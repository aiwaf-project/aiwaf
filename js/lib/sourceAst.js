let parser = null;

try {
  // Optional at runtime for defensive compatibility with older installs.
  // package.json declares this directly for normal installations.
  parser = require('@babel/parser');
} catch (err) {
  parser = null;
}

function parseFunctionSource(handler) {
  if (!parser || typeof handler !== 'function') return null;
  const source = Function.prototype.toString.call(handler);
  const candidates = [
    `(${source})`,
    source
  ];
  for (const candidate of candidates) {
    try {
      return parser.parse(candidate, {
        sourceType: 'unambiguous',
        plugins: [
          'asyncGenerators',
          'classProperties',
          'dynamicImport',
          'jsx',
          'objectRestSpread',
          'optionalChaining',
          'typescript'
        ]
      });
    } catch (err) {
      // Try the next parse shape.
    }
  }
  return null;
}

function walk(node, visitor, parent = null) {
  if (!node || typeof node !== 'object') return;
  if (typeof node.type === 'string') visitor(node, parent);
  for (const [key, value] of Object.entries(node)) {
    if (key === 'loc' || key === 'start' || key === 'end' || key === 'extra') continue;
    if (Array.isArray(value)) {
      value.forEach(child => walk(child, visitor, node));
    } else if (value && typeof value === 'object' && typeof value.type === 'string') {
      walk(value, visitor, node);
    }
  }
}

function propertyName(node) {
  if (!node) return '';
  if (node.type === 'Identifier') return node.name || '';
  if (node.type === 'StringLiteral' || node.type === 'NumericLiteral') return String(node.value);
  return '';
}

function calleeParts(callee) {
  if (!callee) return [];
  if (callee.type === 'Identifier') return [callee.name];
  if (callee.type === 'MemberExpression' || callee.type === 'OptionalMemberExpression') {
    return [...calleeParts(callee.object), propertyName(callee.property)].filter(Boolean);
  }
  return [];
}

function memberParts(node) {
  if (!node || (node.type !== 'MemberExpression' && node.type !== 'OptionalMemberExpression')) return [];
  return [...calleeParts(node.object), propertyName(node.property)].filter(Boolean);
}

function literalString(node) {
  if (!node) return '';
  if (node.type === 'StringLiteral') return String(node.value);
  if (node.type === 'TemplateLiteral' && node.quasis.length === 1) return String(node.quasis[0].value.cooked || '');
  return '';
}

function includesAny(text, tokens) {
  const lower = String(text || '').toLowerCase();
  return tokens.some(token => lower.includes(token));
}

function actionPriority(action) {
  const p = { login: 50, token_login: 50, logout: 40, register: 30, token_auth: 20, password_hashing: 10, user_model: 5 };
  return p[action] || 0;
}

function mergeAction(current, candidate) {
  if (!current) return candidate;
  return actionPriority(candidate) > actionPriority(current) ? candidate : current;
}

function analyzeHandlerAst(handler) {
  const tree = parseFunctionSource(handler);
  if (!tree) {
    return {
      available: false,
      methods: [],
      apiScore: 0,
      apiSignals: [],
      formScore: 0,
      formSignals: [],
      authScore: 0,
      authSignals: [],
      authAction: '',
      requestBody: false,
      payloadType: ''
    };
  }

  const methods = new Set();
  const apiSignals = new Set();
  const formSignals = new Set();
  const authSignals = new Set();
  let apiScore = 0;
  let formScore = 0;
  let authScore = 0;
  let authAction = '';
  let requestBody = false;
  let payloadType = '';
  let sawRenderOrRedirect = false;

  walk(tree, node => {
    if (node.type === 'StringLiteral' || node.type === 'TemplateLiteral') {
      const value = literalString(node);
      const upper = value.toUpperCase();
      if (['GET', 'POST', 'PUT', 'PATCH', 'DELETE'].includes(upper)) methods.add(upper);
      if (value.toLowerCase().includes('application/json')) {
        apiScore += 30;
        payloadType = payloadType || 'json';
        apiSignals.add('content-type:application/json');
      }
    }

    if (node.type === 'BinaryExpression' || node.type === 'LogicalExpression') {
      if (node.left && node.right) {
        const leftParts = memberParts(node.left);
        const rightParts = memberParts(node.right);
        const leftName = leftParts.join('.');
        const rightName = rightParts.join('.');
        const leftLiteral = literalString(node.left);
        const rightLiteral = literalString(node.right);

        if (/^(req|request|ctx\.request)\.method$/.test(leftName) && rightLiteral) {
          methods.add(rightLiteral.toUpperCase());
        }
        if (/^(req|request|ctx\.request)\.method$/.test(rightName) && leftLiteral) {
          methods.add(leftLiteral.toUpperCase());
        }
      }
    }

    if (node.type === 'TSTypeAnnotation' || node.type === 'TypeAnnotation') {
      const typeRef = node.typeAnnotation && node.typeAnnotation.typeName ? node.typeAnnotation.typeName.name : '';
      if (typeRef && !['Request', 'Response', 'NextFunction', 'any', 'void'].includes(typeRef)) {
        apiScore += 25;
        requestBody = true;
        payloadType = payloadType || 'json';
        apiSignals.add(`body_model:${typeRef}`);
      }
    }

    if (node.type === 'ReturnStatement' && node.argument) {
      if (node.argument.type === 'ObjectExpression' || node.argument.type === 'ArrayExpression') {
        apiScore += 50;
        apiSignals.add('return:json_literal');
      }
    }

    if (node.type === 'MemberExpression' || node.type === 'OptionalMemberExpression') {
      const parts = memberParts(node);
      const joined = parts.join('.');
      if (/^(req|request|ctx\.request)\.(body|json|data)$/.test(joined)) {
        apiScore += 20;
        requestBody = true;
        payloadType = payloadType || 'json';
        apiSignals.add(joined);
      }
      if (/^(req|request|ctx\.request)\.(file|files|form)$/.test(joined)) {
        formScore += 45;
        requestBody = true;
        payloadType = 'form';
        formSignals.add(joined);
      }
    }

    if (node.type === 'CallExpression' || node.type === 'OptionalCallExpression') {
      const parts = calleeParts(node.callee);
      const joined = parts.join('.');
      const name = parts[parts.length - 1] || '';
      const nameLower = name.toLowerCase();

      if (['json', 'jsonresponse', 'jsonify'].includes(nameLower) && includesAny(joined, ['res.', 'reply.', 'response.', 'json'])) {
        apiScore += 60;
        apiSignals.add(joined);
      }
      if (['send'].includes(nameLower) && node.arguments?.[0] && ['ObjectExpression', 'ArrayExpression'].includes(node.arguments[0].type)) {
        apiScore += 45;
        apiSignals.add(`${joined}:object`);
      }
      if (['getjson', 'json'].includes(nameLower) && includesAny(joined, ['req.', 'request.', 'ctx.request.'])) {
        apiScore += 20;
        requestBody = true;
        payloadType = payloadType || 'json';
        apiSignals.add(joined);
      }
      if (['render', 'rendertemplate', 'redirect'].includes(nameLower)) {
        sawRenderOrRedirect = true;
        formScore += nameLower.includes('render') ? 40 : 30;
        payloadType = payloadType || 'form';
        formSignals.add(joined || name);
      }

      if (['logout', 'logoutuser', 'signout'].includes(nameLower)) {
        authScore += 60;
        authAction = mergeAction(authAction, 'logout');
        authSignals.add(joined || name);
      }
      if (['createuser', 'register', 'signup'].includes(nameLower)) {
        authScore += 50;
        authAction = mergeAction(authAction, 'register');
        authSignals.add(joined || name);
      }
      if (['authenticate', 'login', 'signin', 'loginuser'].includes(nameLower)) {
        authScore += nameLower.includes('user') || nameLower === 'authenticate' ? 50 : 30;
        authAction = mergeAction(authAction, 'login');
        authSignals.add(joined || name);
      }
      if (['compare', 'comparepassword', 'verify', 'verifypassword', 'checkpasswordhash'].includes(nameLower)) {
        authScore += 35;
        authAction = mergeAction(authAction, 'login');
        authSignals.add(joined || name);
      }
      if (['sign', 'createaccesstoken'].includes(nameLower) || (nameLower === 'create' && includesAny(joined, ['token']))) {
        authScore += 50;
        authAction = mergeAction(authAction, 'token_login');
        authSignals.add(joined || name);
      }
      if (includesAny(joined, ['bcrypt', 'jsonwebtoken', 'jwt', 'passport'])) {
        authScore += 50;
        authAction = mergeAction(authAction, includesAny(joined, ['jwt', 'jsonwebtoken', 'token']) ? 'token_login' : 'login');
        authSignals.add(joined);
      }
    }

    if (node.type === 'Identifier') {
      const nodeLower = node.name.toLowerCase();
      if (['logout', 'signout'].includes(nodeLower)) {
        authScore += 60;
        authAction = mergeAction(authAction, 'logout');
        authSignals.add(node.name);
      } else if (['register', 'signup', 'createuser'].includes(nodeLower)) {
        authScore += 50;
        authAction = mergeAction(authAction, 'register');
        authSignals.add(node.name);
      } else if (includesAny(nodeLower, ['createaccesstoken', 'jwt', 'passport', 'bcrypt'])) {
        authScore += 20;
        authAction = mergeAction(authAction, nodeLower.includes('token') || nodeLower.includes('jwt') ? 'token_login' : 'login');
        authSignals.add(node.name);
      }
    }
  });

  if (sawRenderOrRedirect && requestBody) {
    formScore += 45;
    payloadType = 'form';
    formSignals.add('request.body');
  }

  if (formScore >= 50 && formScore >= apiScore) {
    payloadType = 'form';
  }

  return {
    available: true,
    methods: Array.from(methods).sort(),
    apiScore,
    apiSignals: Array.from(apiSignals).sort(),
    formScore,
    formSignals: Array.from(formSignals).sort(),
    authScore,
    authSignals: Array.from(authSignals).sort(),
    authAction,
    requestBody,
    payloadType
  };
}

module.exports = {
  parseFunctionSource,
  analyzeHandlerAst
};
