// An actions2 action that also reaches for the request itself.
module.exports = {
  inputs: { title: { type: 'string' } },
  fn: async function ({ title }) {
    return { title, note: this.req.body.note };
  }
};
