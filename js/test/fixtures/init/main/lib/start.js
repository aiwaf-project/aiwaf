const express = require('express');

const app = express();
app.get('/main', (req, res) => res.json({ ok: true }));
module.exports = { app };
