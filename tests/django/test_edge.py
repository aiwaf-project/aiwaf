"""EdgeAssistedMiddleware must only trust edge headers from a verified origin."""
from unittest.mock import MagicMock, patch

from django.http import HttpResponse
from django.test import RequestFactory, SimpleTestCase

from aiwaf.django.edge import EdgeAssistedMiddleware

PEER = "198.51.100.7"
EDGE_CLIENT = "203.0.113.5"


class EdgeAssistedMiddlewareTests(SimpleTestCase):
    def setUp(self):
        self.get_response = MagicMock(return_value=HttpResponse("ok"))
        self.factory = RequestFactory()

    def build(self, agent):
        with patch("aiwaf.django.edge.create_agent", return_value=agent):
            return EdgeAssistedMiddleware(self.get_response)

    def request(self, **extra):
        return self.factory.post(
            "/orders/?x=1",
            REMOTE_ADDR=PEER,
            HTTP_CF_CONNECTING_IP=EDGE_CLIENT,
            HTTP_X_BOUVANE_ORIGIN_TOKEN="origin-token",
            **extra,
        )

    def agent(self, trusted=False, blocked=False):
        agent = MagicMock()
        agent.trusted_origin.return_value = trusted
        agent.blocked.return_value = blocked
        return agent

    def test_without_an_agent_requests_pass_through_untouched(self):
        request = self.request()
        response = self.build(None)(request)
        self.assertEqual(response.status_code, 200)
        self.get_response.assert_called_once_with(request)
        self.assertEqual(request.META["HTTP_X_BOUVANE_ORIGIN_TOKEN"], "origin-token")

    def test_agent_starts_at_construction(self):
        agent = self.agent()
        self.build(agent)
        agent.start.assert_called_once_with()

    def test_untrusted_origin_uses_the_peer_address_not_the_edge_header(self):
        agent = self.agent(trusted=False)
        self.build(agent)(self.request())
        self.assertEqual(agent.blocked.call_args.kwargs["ip"], PEER)

    def test_trusted_origin_uses_the_edge_client_address(self):
        agent = self.agent(trusted=True)
        self.build(agent)(self.request())
        self.assertEqual(agent.blocked.call_args.kwargs["ip"], EDGE_CLIENT)

    def test_origin_token_never_reaches_the_application(self):
        agent = self.agent()
        request = self.request()
        self.build(agent)(request)
        agent.trusted_origin.assert_called_once_with("origin-token")
        self.assertEqual(agent.blocked.call_args.kwargs["origin_token"], "origin-token")
        self.assertNotIn("HTTP_X_BOUVANE_ORIGIN_TOKEN", request.META)

    def test_request_details_are_passed_to_the_agent(self):
        agent = self.agent()
        self.build(agent)(self.request())
        kwargs = agent.blocked.call_args.kwargs
        self.assertEqual(kwargs["hostname"], "testserver")
        self.assertEqual(kwargs["path"], "/orders/")
        self.assertEqual(kwargs["method"], "POST")

    def test_blocked_request_gets_a_403_and_skips_the_application(self):
        response = self.build(self.agent(blocked=True))(self.request())
        self.assertEqual(response.status_code, 403)
        self.get_response.assert_not_called()
