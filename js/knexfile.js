module.exports = {
    development: {
      client: 'sqlite3',
      connection: {
        filename: './data/aiwaf.sqlite'
      },
      useNullAsDefault: true
    }
  };