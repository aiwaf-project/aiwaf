// The same, with a switch, and an else for every method the if doesn't name.
export default function handler(req, res) {
  switch (req.method) {
    case 'GET':
    case 'HEAD':
      return res.json({ status: req.query.status });
    case 'DELETE':
      return res.json({ id: req.query.id });
    default:
      break;
  }
  if (req.method === 'PUT') res.json({ note: req.body.note });
  else res.json({ title: req.body.title });
}
