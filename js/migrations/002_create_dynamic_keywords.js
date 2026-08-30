exports.up = function(knex) {
    return knex.schema.createTable('dynamic_keywords', table => {
      table.increments('id');
      table.string('keyword').notNullable().unique();
      table.integer('count').notNullable().defaultTo(0);
      table.timestamp('updated_at').defaultTo(knex.fn.now());
    });
  };
  exports.down = function(knex) {
    return knex.schema.dropTable('dynamic_keywords');
  };