exports.up = function(knex) {
    return knex.schema.createTable('blocked_ips', table => {
      table.increments('id');
      table.string('ip_address').notNullable().unique();
      table.string('reason');
      table.timestamp('created_at').defaultTo(knex.fn.now());
    });
  };
  exports.down = function(knex) {
    return knex.schema.dropTable('blocked_ips');
  };