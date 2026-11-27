// A schema in its own module, as most apps keep them.
module.exports.webhookSchema = {
  body: {
    type: 'object',
    properties: { eventId: { type: 'string', format: 'uuid' }, successful: { type: 'boolean' }, attempt: { type: 'integer' } },
    required: ['eventId', 'successful']
  }
};
