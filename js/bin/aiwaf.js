#!/usr/bin/env node

const path = require('path');
const { spawn } = require('child_process');
const pathManifest = require('../lib/pathManifest');

function print(obj) {
  process.stdout.write(`${JSON.stringify(obj, null, 2)}\n`);
}

async function clearBlacklist() {
  const blacklistManager = require('../lib/blacklistManager');
  const deleted = await blacklistManager.clear();
  print({ ok: true, cleared: deleted });
}

async function clearRequestLogs() {
  const requestLogStore = require('../lib/requestLogStore');
  const deleted = await requestLogStore.clear();
  print({ ok: true, cleared: deleted });
}

function runTrain() {
  const child = spawn(process.execPath, [path.join(__dirname, '..', 'train.js')], {
    stdio: 'inherit',
    env: process.env
  });
  child.on('exit', code => process.exit(code || 0));
}

function readJsonFile(filePath) {
  const fs = require('fs');
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function parseManifestArgs(args = []) {
  const options = {
    framework: 'express',
    output: process.env.AIWAF_PATH_MANIFEST || '.aiwaf/paths.json',
    routes: []
  };
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === '--framework' || arg === '-f') {
      options.framework = args[++index] || options.framework;
    } else if (arg === '--output' || arg === '-o') {
      options.output = args[++index] || options.output;
    } else if (arg === '--routes' || arg === '-r') {
      const routeFile = args[++index];
      options.routes = routeFile ? readJsonFile(routeFile) : [];
    }
  }
  return options;
}

async function diagnoseIp(ip) {
  const blacklistManager = require('../lib/blacklistManager');
  const exemptionStore = require('../lib/exemptionStore');
  const blocked = await blacklistManager.isBlocked(ip);
  const exempt = await exemptionStore.isIpExempt(ip);
  print({ ip, blocked, exempt });
}

function runPathShell() {
  const readline = require('readline');
  const fs = require('fs');
  const manifestPath = process.env.AIWAF_PATH_MANIFEST || '.aiwaf/paths.json';
  if (!fs.existsSync(manifestPath)) {
    console.error(`Manifest not found at ${manifestPath}. Run 'aiwaf manifest' first.`);
    return process.exit(1);
  }
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  const routes = Object.keys(manifest.routes || {});
  
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: 'aiwaf> '
  });
  
  console.log(`Loaded ${routes.length} routes from ${manifestPath}`);
  rl.prompt();
  
  rl.on('line', async (line) => {
    const args = line.trim().split(' ').filter(Boolean);
    if (!args.length) { rl.prompt(); return; }
    const cmd = args[0];
    
    if (cmd === 'exit' || cmd === 'quit') {
      rl.close();
    } else if (cmd === 'ls') {
      routes.forEach((r, i) => console.log(`[${i}] ${r}`));
    } else if (cmd === 'exempt') {
      const idx = Number(args[1]);
      const pathPrefix = !isNaN(idx) && routes[idx] ? routes[idx] : args[1];
      if (!pathPrefix) {
        console.log('Usage: exempt <index|path>');
      } else {
        const exemptionStore = require('../lib/exemptionStore');
        await exemptionStore.addPath(pathPrefix, 'manual via shell');
        console.log(`Exempted path: ${pathPrefix}`);
      }
    } else {
      console.log(`Unknown command: ${cmd}`);
      console.log(`Available commands: ls, exempt <index|path>, exit`);
    }
    rl.prompt();
  }).on('close', () => {
    console.log('Exiting shell.');
    process.exit(0);
  });
}

async function runReset(args) {
  const flags = { blacklist: false, keywords: false, exemptions: false, confirm: false };
  for (const arg of args) {
    if (arg === '--blacklist') flags.blacklist = true;
    if (arg === '--keywords') flags.keywords = true;
    if (arg === '--exemptions') flags.exemptions = true;
    if (arg === '--all') { flags.blacklist = true; flags.keywords = true; flags.exemptions = true; }
    if (arg === '--confirm') flags.confirm = true;
  }
  if (!flags.blacklist && !flags.keywords && !flags.exemptions) {
    return print({ error: 'Specify what to reset: --blacklist, --keywords, --exemptions, or --all' });
  }
  if (!flags.confirm) {
    return print({ error: 'Must pass --confirm to execute reset' });
  }

  const results = {};
  if (flags.blacklist) {
    const blacklistManager = require('../lib/blacklistManager');
    results.blacklist = await blacklistManager.clear();
  }
  if (flags.keywords) {
    const dynamicKeywordStore = require('../lib/dynamicKeywordStore');
    results.keywords = await dynamicKeywordStore.clear();
  }
  if (flags.exemptions) {
    const exemptionStore = require('../lib/exemptionStore');
    results.exemptions = await exemptionStore.clear();
  }
  return print({ ok: true, reset: results });
}

async function main() {
  const [, , cmd, subcmd, ...args] = process.argv;

  if (!cmd || ['help', '--help', '-h'].includes(cmd)) {
    print({
      usage: 'aiwaf <command> [subcommand] [args]',
      commands: [
        'list blacklist|exemptions|geo|request-logs',
        'list dynamic-keywords',
        'list model-info',
        'add blacklist <ip> [reason]',
        'add ip-exemption <ip> [reason]',
        'add path-exemption <pathPrefix> [reason]',
        'add dynamic-keyword <keyword> [count]',
        'remove blacklist <ip>',
        'remove ip-exemption <ip>',
        'remove path-exemption <pathPrefix>',
        'remove dynamic-keyword <keyword>',
        'geo block <CC> [reason]',
        'geo unblock <CC>',
        'geo summary',
        'clear blacklist|request-logs',
        'clear dynamic-keywords',
        'train',
        'manifest --framework <name> --routes <routes.json> [--output .aiwaf/paths.json]',
        'whois <domain|ip>',
        'diagnose <ip>',
        'reset [--blacklist] [--keywords] [--exemptions] [--all] --confirm',
        'pathshell'
      ]
    });
    return;
  }

  if (cmd === 'train') return runTrain();
  if (cmd === 'pathshell') return runPathShell();
  if (cmd === 'reset') return runReset([subcmd, ...args].filter(Boolean));

  if (cmd === 'manifest') {
    const options = parseManifestArgs([subcmd, ...args].filter(Boolean));
    const manifest = pathManifest.generateFrameworkManifest(
      options.framework,
      null,
      options.output,
      { routes: options.routes }
    );
    return print({ ok: true, output: options.output, framework: options.framework, routes: Object.keys(manifest.routes || {}).length });
  }

  if (cmd === 'list' && subcmd === 'blacklist') {
    const blacklistManager = require('../lib/blacklistManager');
    const rows = await blacklistManager.getBlockedIPs();
    return print(rows);
  }

  if (cmd === 'list' && subcmd === 'exemptions') {
    const exemptionStore = require('../lib/exemptionStore');
    const ips = await exemptionStore.listIps();
    const paths = await exemptionStore.listPaths();
    return print({ ip_exemptions: ips, path_exemptions: paths });
  }

  if (cmd === 'list' && subcmd === 'geo') {
    const geoStore = require('../lib/geoStore');
    const rows = await geoStore.listBlockedCountries();
    return print(rows);
  }

  if (cmd === 'list' && subcmd === 'request-logs') {
    const requestLogStore = require('../lib/requestLogStore');
    const limit = Number(args[0] || 100);
    const rows = await requestLogStore.recent(limit);
    return print(rows);
  }

  if (cmd === 'list' && subcmd === 'dynamic-keywords') {
    const dynamicKeywordStore = require('../lib/dynamicKeywordStore');
    const limit = Number(args[0] || 100);
    const rows = await dynamicKeywordStore.list(limit);
    return print(rows);
  }

  if (cmd === 'list' && subcmd === 'model-info') {
    const modelStore = require('../lib/modelStore');
    const model = await modelStore.load(process.env);
    if (!model) return print({ loaded: false });
    return print({ loaded: true, metadata: model.metadata || null });
  }

  if (cmd === 'add' && subcmd === 'ip-exemption') {
    const exemptionStore = require('../lib/exemptionStore');
    const [ip, ...reasonParts] = args;
    await exemptionStore.addIp(ip, reasonParts.join(' ') || 'manual');
    return print({ ok: true, ip });
  }

  if (cmd === 'add' && subcmd === 'blacklist') {
    const blacklistManager = require('../lib/blacklistManager');
    const [ip, ...reasonParts] = args;
    await blacklistManager.block(ip, reasonParts.join(' ') || 'manual');
    return print({ ok: true, ip });
  }

  if (cmd === 'add' && subcmd === 'path-exemption') {
    const exemptionStore = require('../lib/exemptionStore');
    const [pathPrefix, ...reasonParts] = args;
    await exemptionStore.addPath(pathPrefix, reasonParts.join(' ') || 'manual');
    return print({ ok: true, pathPrefix });
  }

  if (cmd === 'add' && subcmd === 'dynamic-keyword') {
    const dynamicKeywordStore = require('../lib/dynamicKeywordStore');
    const [keyword, count] = args;
    await dynamicKeywordStore.add(keyword, Number(count || 1));
    return print({ ok: true, keyword });
  }

  if (cmd === 'remove' && subcmd === 'blacklist') {
    const blacklistManager = require('../lib/blacklistManager');
    const [ip] = args;
    const deleted = await blacklistManager.unblock(ip);
    return print({ ok: true, deleted });
  }

  if (cmd === 'remove' && subcmd === 'ip-exemption') {
    const exemptionStore = require('../lib/exemptionStore');
    const [ip] = args;
    const deleted = await exemptionStore.removeIp(ip);
    return print({ ok: true, deleted });
  }

  if (cmd === 'remove' && subcmd === 'dynamic-keyword') {
    const dynamicKeywordStore = require('../lib/dynamicKeywordStore');
    const [keyword] = args;
    await dynamicKeywordStore.remove(keyword);
    return print({ ok: true, keyword });
  }

  if (cmd === 'remove' && subcmd === 'path-exemption') {
    const exemptionStore = require('../lib/exemptionStore');
    const [pathPrefix] = args;
    const deleted = await exemptionStore.removePath(pathPrefix);
    return print({ ok: true, deleted });
  }

  if (cmd === 'geo' && subcmd === 'block') {
    const geoStore = require('../lib/geoStore');
    const [countryCode, ...reasonParts] = args;
    await geoStore.addBlockedCountry(countryCode, reasonParts.join(' ') || 'manual');
    return print({ ok: true, countryCode });
  }

  if (cmd === 'geo' && subcmd === 'unblock') {
    const geoStore = require('../lib/geoStore');
    const [countryCode] = args;
    const deleted = await geoStore.removeBlockedCountry(countryCode);
    return print({ ok: true, deleted });
  }

  if (cmd === 'geo' && subcmd === 'summary') {
    const requestLogStore = require('../lib/requestLogStore');
    const rows = await requestLogStore.geoSummary(50);
    return print(rows);
  }

  if (cmd === 'clear' && subcmd === 'blacklist') {
    return clearBlacklist();
  }

  if (cmd === 'clear' && subcmd === 'request-logs') {
    return clearRequestLogs();
  }

  if (cmd === 'clear' && subcmd === 'dynamic-keywords') {
    const dynamicKeywordStore = require('../lib/dynamicKeywordStore');
    const cleared = await dynamicKeywordStore.clear();
    return print({ ok: true, cleared });
  }

  if (cmd === 'diagnose') {
    const [ip] = [subcmd, ...args];
    return diagnoseIp(ip);
  }

  if (cmd === 'whois') {
    const whois = require('../lib/whois');
    const [target] = [subcmd, ...args];
    const result = await whois.runWhoisLookup(target);
    process.stdout.write(`${result}\n`);
    return;
  }

  print({ error: 'unknown command', command: [cmd, subcmd, ...args].join(' ') });
  process.exitCode = 1;
}

main()
  .catch(err => {
    process.stderr.write(`${err.stack || err.message}\n`);
    process.exit(1);
  })
  .finally(async () => {
    // no-op
  });
