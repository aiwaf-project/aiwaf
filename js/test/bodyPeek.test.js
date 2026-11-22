const { PassThrough } = require('stream');
const { peekBody, isMultipart } = require('../lib/bodyPeek');

const stream = headers => Object.assign(new PassThrough(), { headers });

test('reads the start of a body and puts it back for the next reader', async () => {
  const req = stream({ 'content-length': '20' });
  req.end(Buffer.from('a'.repeat(20)));
  const prefix = await peekBody(req, 8);
  expect(prefix.length).toBeGreaterThan(8);
  const rest = [];
  for await (const chunk of req) rest.push(chunk);
  expect(Buffer.concat(rest).toString()).toBe('a'.repeat(20));
});

test('a whole body shorter than the limit is read and put back', async () => {
  const req = stream({});
  req.end('short');
  expect((await peekBody(req, 1024)).toString()).toBe('short');
});

test('a failing stream rejects', async () => {
  const req = stream({});
  const peeked = peekBody(req, 8);
  req.destroy(new Error('gone'));
  await expect(peeked).rejects.toThrow('gone');
});

test('multipart forms are recognised by their content type', () => {
  expect(isMultipart({ headers: { 'content-type': 'multipart/form-data; boundary=x' } })).toBe(true);
  expect(isMultipart({ headers: { 'content-type': 'application/json' } })).toBe(false);
  expect(isMultipart({})).toBe(false);
});
