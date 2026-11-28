// A Pages API route serving GET and POST from one handler, by branching on req.method.
export default function handler(req, res) {
  const trace = req.query.trace;
  if (req.method === 'GET') {
    return res.json({ q: req.query.q, page: parseInt(req.query.page, 10), trace });
  }
  if (req.method !== 'POST') return res.status(405).end();
  return res.json({ title: req.body.title, page: req.body.page });
}
