import pytest
from aiwaf.flask.db_models import WhitelistedIP, BlacklistedIP, Keyword, db

def test_whitelisted_ip_creation(app_context):
    """Test creating a whitelisted IP."""
    ip_entry = WhitelistedIP(ip='192.168.1.1')
    db.session.add(ip_entry)
    db.session.commit()
    
    assert ip_entry.id is not None
    assert ip_entry.ip == '192.168.1.1'

def test_whitelisted_ip_unique_constraint(app_context):
    """Test that duplicate IPs are not allowed in whitelist."""
    ip1 = WhitelistedIP(ip='192.168.1.1')
    ip2 = WhitelistedIP(ip='192.168.1.1')
    
    db.session.add(ip1)
    db.session.commit()
    
    db.session.add(ip2)
    with pytest.raises(Exception):  # IntegrityError
        db.session.commit()

def test_blacklisted_ip_creation(app_context):
    """Test creating a blacklisted IP."""
    ip_entry = BlacklistedIP(
        ip='10.0.0.1',
        reason='suspicious activity',
        extended_request_info={"path": "/login", "method": "POST"},
    )
    db.session.add(ip_entry)
    db.session.commit()
    
    assert ip_entry.id is not None
    assert ip_entry.ip == '10.0.0.1'
    assert ip_entry.reason == 'suspicious activity'
    assert ip_entry.extended_request_info["path"] == "/login"

def test_blacklisted_ip_unique_constraint(app_context):
    """Test that duplicate IPs are not allowed in blacklist."""
    ip1 = BlacklistedIP(ip='10.0.0.1', reason='test')
    ip2 = BlacklistedIP(ip='10.0.0.1', reason='test2')
    
    db.session.add(ip1)
    db.session.commit()
    
    db.session.add(ip2)
    with pytest.raises(Exception):  # IntegrityError
        db.session.commit()

def test_keyword_creation(app_context):
    """Test creating a keyword."""
    keyword = Keyword(keyword='malicious')
    db.session.add(keyword)
    db.session.commit()
    
    assert keyword.id is not None
    assert keyword.keyword == 'malicious'

def test_keyword_unique_constraint(app_context):
    """Test that duplicate keywords are not allowed."""
    kw1 = Keyword(keyword='malicious')
    kw2 = Keyword(keyword='malicious')
    
    db.session.add(kw1)
    db.session.commit()
    
    db.session.add(kw2)
    with pytest.raises(Exception):  # IntegrityError
        db.session.commit()
