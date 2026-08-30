const mockDynamicKeywordStore = {
  initialize: jest.fn(async () => {}),
  list: jest.fn(async () => [{ keyword: 'admin', count: 5 }]),
  increment: jest.fn(async () => {}),
  clear: jest.fn(async () => 0)
};

jest.mock('../lib/dynamicKeywordStore', () => mockDynamicKeywordStore);

const dynamicKeyword = require('../lib/dynamicKeyword');

describe('dynamicKeyword integration', () => {
  it('hydrates counts from store and increments on learn', async () => {
    dynamicKeyword.init({ dynamicTopN: 3 });

    await new Promise(resolve => setTimeout(resolve, 0));

    const match = dynamicKeyword.check('/admin/settings');
    expect(match).toBe('admin');

    dynamicKeyword.learn('/admin/dashboard');
    expect(mockDynamicKeywordStore.increment).toHaveBeenCalledWith('admin');
  });
});
