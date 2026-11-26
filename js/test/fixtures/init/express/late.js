// Starts after an asynchronous step, as an app that connects to a database first.
const express = require('express');

const app = express();
app.get('/late', (req, res) => res.json({ ok: true }));
setTimeout(() => app.listen(0), 50);
