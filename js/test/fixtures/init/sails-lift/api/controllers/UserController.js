// A classic Sails controller: each action takes req and res.
module.exports = {
  find: async function (req, res) {
    return res.json({ q: req.param('q'), page: parseInt(req.query.page, 10) });
  },
  create: async function (req, res) {
    return res.json({ name: req.body.name, age: Number(req.body.age) });
  },
  async search(req, res) {
    return res.json(req.allParams());
  }
};
