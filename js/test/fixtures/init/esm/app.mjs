import express from 'express';

const app = express();
app.get('/esm', (req, res) => res.json({ ok: true }));
export default app;
