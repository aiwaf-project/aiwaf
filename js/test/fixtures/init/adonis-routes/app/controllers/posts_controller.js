// Another controller with an index method: each links to its own route.
export default class PostsController {
  async index({ request }) {
    return { tag: request.qs().tag };
  }
}
