describe('redisClient', () => {
  beforeEach(() => {
    jest.resetModules();
    delete process.env.REDIS_URL;
    delete process.env.AIWAF_REDIS_URL;
  });

  it('connects using AIWAF_REDIS_URL when REDIS_URL is missing', async () => {
    process.env.AIWAF_REDIS_URL = 'redis://localhost:6379';

    jest.doMock('redis', () => ({
      createClient: jest.fn(() => {
        const client = {
          isOpen: true,
          on: jest.fn(),
          connect: jest.fn(async () => {})
        };
        return client;
      })
    }));

    let client;
    await new Promise(resolve => {
      jest.isolateModules(() => {
        const rc = require('../lib/redisClient');
        rc.connect().then(() => {
          client = rc.getClient();
          resolve();
        });
      });
    });

    expect(client).toBeTruthy();
  });
});
