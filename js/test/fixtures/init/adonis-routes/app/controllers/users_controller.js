// An AdonisJS controller: each method destructures the request from the HTTP context.
export default class UsersController {
  async index({ request, response }) {
    const { page, q } = request.qs();
    return response.json({ page: Number(page), q, sort: request.input('sort') });
  }

  async store({ request }) {
    const body = request.body();
    return { email: body.email, ...request.only(['name', 'age']) };
  }

  async update({ request }) {
    return request.all();
  }
}
