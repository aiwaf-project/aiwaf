const SCANNING_PATTERNS = [
  'wp-admin', 'wp-content', 'wp-includes', 'wp-config', 'xmlrpc.php',
  'admin', 'phpmyadmin', 'adminer', 'config', 'configuration',
  'settings', 'setup', 'install', 'installer',
  'backup', 'database', 'db', 'mysql', 'sql', 'dump',
  '.env', '.git', '.htaccess', '.htpasswd', 'passwd', 'shadow',
  'cgi-bin', 'scripts', 'shell', 'cmd', 'exec',
  '.php', '.asp', '.aspx', '.jsp', '.cgi', '.pl'
];

function isScanningPath(requestPath) {
  const pathLower = String(requestPath || '').toLowerCase();
  if (SCANNING_PATTERNS.some(pattern => pathLower.includes(pattern))) return true;
  if (pathLower.includes('../') || pathLower.includes('..\\')) return true;
  return ['%2e%2e', '%252e', '%c0%ae'].some(encoded => pathLower.includes(encoded));
}

function getDefaultLegitimateKeywords() {
  return new Set([
    'profile', 'user', 'users', 'account', 'accounts', 'settings', 'dashboard',
    'home', 'about', 'contact', 'help', 'search', 'list', 'lists',
    'view', 'views', 'edit', 'create', 'update', 'delete', 'detail', 'details',
    'api', 'auth', 'login', 'logout', 'register', 'signup', 'signin',
    'reset', 'confirm', 'activate', 'verify', 'page', 'pages',
    'category', 'categories', 'tag', 'tags', 'post', 'posts',
    'article', 'articles', 'blog', 'blogs', 'news', 'item', 'items',
    'admin', 'administration', 'manage', 'manager', 'control', 'panel',
    'config', 'configuration', 'option', 'options', 'preference', 'preferences',
    'contenttypes', 'contenttype', 'sessions', 'session', 'messages', 'message',
    'staticfiles', 'static', 'sites', 'site', 'flatpages', 'flatpage',
    'redirects', 'redirect', 'permissions', 'permission', 'groups', 'group',
    'token', 'tokens', 'oauth', 'social', 'rest', 'framework', 'cors',
    'debug', 'toolbar', 'extensions', 'allauth', 'crispy', 'forms',
    'channels', 'celery', 'redis', 'cache', 'email', 'mail',
    'favicon', 'robots', 'sitemap', 'manifest', 'health', 'ping',
    'status', 'metrics', 'test', 'docs', 'documentation',
    'endpoint', 'endpoints', 'resource', 'resources', 'data', 'export',
    'import', 'upload', 'download', 'file', 'files', 'media', 'images',
    'documents', 'reports', 'analytics', 'stats', 'statistics',
    'customer', 'customers', 'client', 'clients', 'company', 'companies',
    'department', 'departments', 'employee', 'employees', 'team', 'teams',
    'project', 'projects', 'task', 'tasks', 'event', 'events',
    'notification', 'notifications', 'alert', 'alerts',
    'language', 'languages', 'locale', 'locales', 'translation', 'translations',
    'en', 'fr', 'de', 'es', 'it', 'pt', 'ru', 'ja', 'zh', 'ko'
  ]);
}

function isMaliciousContext(requestPath, keyword, status, staticKeywords = [], pathExistsFn = null) {
  try {
    if (pathExistsFn && pathExistsFn(requestPath)) return false;
  } catch (err) {
    // Path existence checks are best-effort.
  }

  const pathLower = String(requestPath || '').toLowerCase();
  const segments = pathLower.split(/\W+/);
  const staticKeywordSet = new Set(staticKeywords);
  const staticHits = segments.filter(segment => staticKeywordSet.has(segment)).length;

  return staticHits > 1
    || [
      '../', '..\\', '.env', 'wp-admin', 'phpmyadmin', 'config',
      'backup', 'database', 'mysql', 'passwd', 'shadow', 'xmlrpc',
      'shell', 'cmd', 'exec', 'eval', 'system'
    ].some(pattern => pathLower.includes(pattern))
    || [
      'union+select', 'drop+table', '<script', 'javascript:',
      '${', '{{', 'onload=', 'onerror=', 'file://', 'http://'
    ].some(pattern => pathLower.includes(pattern))
    || pathLower.split('../').length - 1 > 1
    || pathLower.split('..\\').length - 1 > 1
    || ['%2e%2e', '%252e', '%c0%ae', '%3c%73%63%72%69%70%74'].some(encoded => pathLower.includes(encoded))
    || (String(status) === '404' && (
      pathLower.length > 50
      || pathLower.split('/').length - 1 > 10
      || ['<', '>', '{', '}', '$', '`'].some(char => pathLower.includes(char))
    ))
    || !!keyword && pathLower.includes(String(keyword).toLowerCase()) && isScanningPath(pathLower);
}

module.exports = {
  SCANNING_PATTERNS,
  isScanningPath,
  getDefaultLegitimateKeywords,
  isMaliciousContext
};
