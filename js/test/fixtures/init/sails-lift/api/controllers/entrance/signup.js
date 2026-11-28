// An actions2 action: Sails passes fn only its declared inputs.
module.exports = {
  inputs: {
    email: { type: 'string', required: true },
    age: { type: 'number', isInteger: true },
    newsletter: { type: 'boolean' }
  },
  exits: { success: { statusCode: 201 } },
  fn: async function ({ email, age, newsletter }) {
    return { email, age, newsletter };
  }
};
