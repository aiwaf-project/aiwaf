// Express: each method's handler keeps its own inputs and types, and a name read from the query
// and the body keeps each location's type.
const express = require('express');
const app = express();
app.use(express.json());
app.get('/items', (req, res) => res.json({ q: req.query.q, page: parseInt(req.query.page, 10) }));
app.post('/items', (req, res) => res.json({ title: req.body.title, page: req.body.page }));
app.patch('/items', (req, res) => res.json({ note: req.body.note }));
app.post('/search', (req, res) => res.json({ limit: Number.parseInt(req.query.limit, 10), label: req.body.limit }));
app.get('/flags', (req, res) => res.json({ on: Boolean(req.query.archived), mode: req.query.mode === 'true' }));
module.exports = app;
// Each line's keys, from iterating the array, end up as the route's nested keys.
app.post('/orders', (req, res) => res.json(req.body.lines.map(line => ({ sku: line.sku, qty: Number(line.qty) }))));
