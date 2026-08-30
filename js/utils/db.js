const knex = require('knex');

const isTest = process.env.NODE_ENV === 'test';

module.exports = knex({
  client: 'sqlite3',
  connection: {
    filename: isTest ? ':memory:' : './aiwaf.sqlite'
  },
  useNullAsDefault: true,
});