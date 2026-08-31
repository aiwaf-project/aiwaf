from aiwaf.flask.models import IPExemption


def test_ip_exemption_model_stub_stores_ip():
    assert IPExemption("203.0.113.1").ip == "203.0.113.1"
