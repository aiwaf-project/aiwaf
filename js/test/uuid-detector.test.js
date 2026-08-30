const uuidDetector = require('../lib/uuidDetector');

describe('uuidDetector', () => {
  it('flags non-uuid path segments', async () => {
    uuidDetector.init({ uuidRoutePrefix: '/user' });
    const req = { path: '/user/not-a-uuid' };
    const result = await uuidDetector.isSuspicious(req);
    expect(result).toBe(true);
  });

  it('uses resolver when provided', async () => {
    uuidDetector.init({
      uuidRoutePrefix: '/user',
      uuidResolver: async (uuid) => uuid !== '00000000-0000-0000-0000-000000000000'
    });
    const req = { path: '/user/00000000-0000-0000-0000-000000000000' };
    const result = await uuidDetector.isSuspicious(req);
    expect(result).toBe(true);
  });
});
