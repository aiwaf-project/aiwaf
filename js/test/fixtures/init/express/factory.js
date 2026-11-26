const express = require('express');

module.exports.createApp = async () => {
  const app = express();
  app.get('/built', (req, res) => res.json({ ok: true }));
  return app;
};
