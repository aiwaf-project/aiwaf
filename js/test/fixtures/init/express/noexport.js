// Exports nothing: the app is found through the server app.listen() starts.
const express = require('express');

const app = express();
app.get('/only', (req, res) => res.json({ ok: true }));
app.listen(0);
