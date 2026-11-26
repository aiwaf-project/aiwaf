const express = require('express');

const router = express.Router();
router.get('/profile', (req, res) => res.json({ user: req.query.user }));
router.post('/login', (req, res) => {
  const { username, password } = req.body;
  res.json({ ok: Boolean(username && password) });
});
module.exports = router;
