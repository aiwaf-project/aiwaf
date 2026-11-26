// One router mounted at two prefixes, another mounted twice, and one with a parameter prefix.
const express = require('express');

const app = express();
const profile = express.Router();
profile.get('/profile', (req, res) => res.json({ ok: true }));
app.use(['/account', '/me'], profile);
const shared = express.Router();
shared.get('/list', (req, res) => res.json([]));
app.use('/one', shared);
app.use('/two', shared);
const posts = express.Router({ mergeParams: true });
posts.get('/posts', (req, res) => res.json({ user: req.params.user || req.params.member }));
app.use(['/users/:user', '/members/:member'], posts);
module.exports = app;
