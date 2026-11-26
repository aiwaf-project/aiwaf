// Seven routes, two on a router mounted at /account; /orders is JSON outside /api.
const express = require('express');
const account = require('./routes/account');

const app = express();
app.use(express.json());
app.get('/', (req, res) => res.send('home'));
app.get('/health', (req, res) => res.json({ ok: true }));
function createOrder(req, res) {
  const { sku, quantity } = req.body;
  res.json({ sku, quantity: Number(quantity) });
}
app.post('/orders', createOrder);
app.get('/api/items/:id', (req, res) => res.json({ id: req.params.id, verbose: req.query.verbose === 'true' }));
app.post('/contact', (req, res) => {
  const { email } = req.body;
  res.redirect(`/thanks?e=${encodeURIComponent(email)}`);
});
app.use('/account', account);
global.aiwafFixtureServer = app.listen(0, () => { global.aiwafFixtureListened = true; });
module.exports = app;
