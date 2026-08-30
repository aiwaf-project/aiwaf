const path = require('path');
const fs = require('fs');
const os = require('os');

function setupMocks() {
  jest.doMock('../lib/blacklistManager', () => ({
    getBlockedIPs: jest.fn(async () => [{ ip_address: '203.0.113.1' }]),
    clear: jest.fn(async () => 3),
    isBlocked: jest.fn(async () => true),
    block: jest.fn(async () => true),
    unblock: jest.fn(async () => true)
  }));

  jest.doMock('../lib/exemptionStore', () => ({
    listIps: jest.fn(async () => [{ ip_address: '198.51.100.1' }]),
    listPaths: jest.fn(async () => [{ path_prefix: '/health' }]),
    addIp: jest.fn(async () => {}),
    addPath: jest.fn(async () => {}),
    removeIp: jest.fn(async () => 1),
    removePath: jest.fn(async () => 1),
    isIpExempt: jest.fn(async () => false)
  }));

  jest.doMock('../lib/geoStore', () => ({
    listBlockedCountries: jest.fn(async () => [{ country_code: 'CN' }]),
    addBlockedCountry: jest.fn(async () => {}),
    removeBlockedCountry: jest.fn(async () => 1)
  }));

  jest.doMock('../lib/requestLogStore', () => ({
    recent: jest.fn(async () => [{ path: '/x' }]),
    geoSummary: jest.fn(async () => [{ country: 'US', requests: 1 }]),
    clear: jest.fn(async () => 9)
  }));

  jest.doMock('../lib/modelStore', () => ({
    load: jest.fn(async () => ({ metadata: { createdAt: 'now' } }))
  }));

  jest.doMock('../lib/dynamicKeywordStore', () => ({
    list: jest.fn(async () => [{ keyword: 'probe', count: 2 }]),
    add: jest.fn(async () => {}),
    remove: jest.fn(async () => {}),
    clear: jest.fn(async () => 1)
  }));

  jest.doMock('../lib/pathManifest', () => ({
    generateFrameworkManifest: jest.fn(() => ({ routes: { '/api/test': {} } }))
  }));

  const on = jest.fn((event, cb) => {
    if (event === 'exit') cb(0);
  });

  jest.doMock('child_process', () => ({
    spawn: jest.fn(() => ({ on }))
  }));

  return {
    blacklistManager: require('../lib/blacklistManager'),
    exemptionStore: require('../lib/exemptionStore'),
    geoStore: require('../lib/geoStore'),
    requestLogStore: require('../lib/requestLogStore'),
    modelStore: require('../lib/modelStore'),
    dynamicKeywordStore: require('../lib/dynamicKeywordStore'),
    pathManifest: require('../lib/pathManifest'),
    childProcess: require('child_process')
  };
}

describe('aiwaf CLI routing', () => {
  let stdoutSpy;
  let stderrSpy;
  let exitSpy;

  beforeEach(() => {
    jest.resetModules();
    stdoutSpy = jest.spyOn(process.stdout, 'write').mockImplementation(() => true);
    stderrSpy = jest.spyOn(process.stderr, 'write').mockImplementation(() => true);
    exitSpy = jest.spyOn(process, 'exit').mockImplementation(() => undefined);
  });

  afterEach(() => {
    stdoutSpy.mockRestore();
    stderrSpy.mockRestore();
    exitSpy.mockRestore();
  });

  function runWithArgv(argv) {
    process.argv = argv;
    jest.isolateModules(() => {
      require('../bin/aiwaf.js');
    });
  }

  it('routes list blacklist', async () => {
    const mocks = setupMocks();
    runWithArgv(['node', path.join('bin', 'aiwaf.js'), 'list', 'blacklist']);

    await new Promise(resolve => setTimeout(resolve, 0));
    expect(mocks.blacklistManager.getBlockedIPs).toHaveBeenCalled();
  });

  it('routes add/remove exemptions and geo commands', async () => {
    const mocks = setupMocks();
    runWithArgv(['node', path.join('bin', 'aiwaf.js'), 'add', 'ip-exemption', '198.51.100.9', 'ops']);
    runWithArgv(['node', path.join('bin', 'aiwaf.js'), 'remove', 'path-exemption', '/health']);
    runWithArgv(['node', path.join('bin', 'aiwaf.js'), 'geo', 'block', 'RU', 'manual']);

    await new Promise(resolve => setTimeout(resolve, 0));
    expect(mocks.exemptionStore.addIp).toHaveBeenCalled();
    expect(mocks.exemptionStore.removePath).toHaveBeenCalled();
    expect(mocks.geoStore.addBlockedCountry).toHaveBeenCalled();
  });

  it('routes blacklist and dynamic keyword commands', async () => {
    const mocks = setupMocks();
    runWithArgv(['node', path.join('bin', 'aiwaf.js'), 'add', 'blacklist', '203.0.113.9', 'manual']);
    runWithArgv(['node', path.join('bin', 'aiwaf.js'), 'remove', 'blacklist', '203.0.113.9']);
    runWithArgv(['node', path.join('bin', 'aiwaf.js'), 'add', 'dynamic-keyword', 'probe', '4']);
    runWithArgv(['node', path.join('bin', 'aiwaf.js'), 'remove', 'dynamic-keyword', 'probe']);

    await new Promise(resolve => setTimeout(resolve, 0));
    expect(mocks.blacklistManager.block).toHaveBeenCalledWith('203.0.113.9', 'manual');
    expect(mocks.blacklistManager.unblock).toHaveBeenCalledWith('203.0.113.9');
    expect(mocks.dynamicKeywordStore.add).toHaveBeenCalledWith('probe', 4);
    expect(mocks.dynamicKeywordStore.remove).toHaveBeenCalledWith('probe');
  });

  it('routes train command to child_process.spawn', async () => {
    const mocks = setupMocks();
    runWithArgv(['node', path.join('bin', 'aiwaf.js'), 'train']);

    await new Promise(resolve => setTimeout(resolve, 0));
    expect(mocks.childProcess.spawn).toHaveBeenCalled();
  });

  it('routes clear request-logs and diagnose', async () => {
    const mocks = setupMocks();
    runWithArgv(['node', path.join('bin', 'aiwaf.js'), 'clear', 'request-logs']);
    runWithArgv(['node', path.join('bin', 'aiwaf.js'), 'diagnose', '198.51.100.2']);

    await new Promise(resolve => setTimeout(resolve, 0));
    expect(mocks.requestLogStore.clear).toHaveBeenCalled();
    expect(mocks.blacklistManager.isBlocked).toHaveBeenCalled();
    expect(mocks.exemptionStore.isIpExempt).toHaveBeenCalled();
  });

  it('routes list model-info', async () => {
    const mocks = setupMocks();
    runWithArgv(['node', path.join('bin', 'aiwaf.js'), 'list', 'model-info']);

    await new Promise(resolve => setTimeout(resolve, 0));
    expect(mocks.modelStore.load).toHaveBeenCalled();
  });

  it('routes manifest generation', async () => {
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'aiwaf-cli-test-'));
    const routesFile = path.join(tempDir, 'routes.json');
    const output = path.join(tempDir, 'paths.json');
    fs.writeFileSync(routesFile, JSON.stringify([{ method: 'GET', path: '/api/test' }]));

    const mocks = setupMocks();
    runWithArgv([
      'node',
      path.join('bin', 'aiwaf.js'),
      'manifest',
      '--framework',
      'fastify',
      '--routes',
      routesFile,
      '--output',
      output
    ]);

    await new Promise(resolve => setTimeout(resolve, 0));
    expect(mocks.pathManifest.generateFrameworkManifest).toHaveBeenCalledWith(
      'fastify',
      null,
      output,
      { routes: [{ method: 'GET', path: '/api/test' }] }
    );
  });
});
