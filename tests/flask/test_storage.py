import pytest
import tempfile
import shutil
from pathlib import Path
from aiwaf.flask.storage import (
    is_ip_whitelisted, add_ip_whitelist,
    is_ip_blacklisted, add_ip_blacklist, remove_ip_blacklist,
    add_keyword, get_top_keywords,
    add_geo_blocked_country, remove_geo_blocked_country,
    is_country_geo_blocked, get_geo_blocked_countries
)

def test_add_ip_whitelist(app_context):
    """Test adding IP to whitelist."""
    ip = '192.168.1.1'
    assert not is_ip_whitelisted(ip)
    
    add_ip_whitelist(ip)
    assert is_ip_whitelisted(ip)

def test_add_duplicate_ip_whitelist(app_context):
    """Test adding duplicate IP to whitelist."""
    ip = '192.168.1.1'
    add_ip_whitelist(ip)
    add_ip_whitelist(ip)  # Should not raise error
    assert is_ip_whitelisted(ip)

def test_add_ip_blacklist(app_context):
    """Test adding IP to blacklist."""
    ip = '10.0.0.1'
    reason = 'suspicious activity'
    assert not is_ip_blacklisted(ip)
    
    add_ip_blacklist(ip, reason)
    assert is_ip_blacklisted(ip)

def test_remove_ip_blacklist(app_context):
    """Test removing IP from blacklist."""
    ip = '10.0.0.1'
    add_ip_blacklist(ip, 'test')
    assert is_ip_blacklisted(ip)
    
    remove_ip_blacklist(ip)
    assert not is_ip_blacklisted(ip)

def test_remove_nonexistent_ip_blacklist(app_context):
    """Test removing non-existent IP from blacklist."""
    ip = '10.0.0.1'
    remove_ip_blacklist(ip)  # Should not raise error
    assert not is_ip_blacklisted(ip)

def test_add_keyword(app_context):
    """Test adding keyword."""
    keyword = 'malicious'
    add_keyword(keyword)
    keywords = get_top_keywords()
    assert keyword in keywords

def test_add_duplicate_keyword(app_context):
    """Test adding duplicate keyword."""
    keyword = 'malicious'
    add_keyword(keyword)
    add_keyword(keyword)  # Should not raise error
    keywords = get_top_keywords()
    assert keyword in keywords

def test_get_top_keywords(app_context):
    """Test getting top keywords."""
    keywords = ['kw1', 'kw2', 'kw3']
    for kw in keywords:
        add_keyword(kw)
    
    top_keywords = get_top_keywords(2)
    assert len(top_keywords) == 2
    for kw in top_keywords:
        assert kw in keywords

def test_geo_blocked_countries_db(app_context):
    """Test geo blocked countries in database storage."""
    country = 'US'
    assert not is_country_geo_blocked(country)

    add_geo_blocked_country(country)
    assert is_country_geo_blocked(country)
    assert country in get_geo_blocked_countries()

    remove_geo_blocked_country(country)
    assert not is_country_geo_blocked(country)

# CSV-specific tests
@pytest.fixture
def csv_app():
    """Create Flask app configured for CSV storage."""
    from flask import Flask
    
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.config['AIWAF_USE_CSV'] = True
    
    # Create temporary directory for CSV files
    temp_dir = tempfile.mkdtemp()
    app.config['AIWAF_DATA_DIR'] = temp_dir
    
    yield app
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def csv_app_context(csv_app):
    """Create application context for CSV storage."""
    with csv_app.app_context():
        yield csv_app

def test_csv_storage_whitelist(csv_app_context):
    """Test CSV storage for whitelist operations."""
    ip = '192.168.100.1'
    
    # Test add and check
    add_ip_whitelist(ip)
    assert is_ip_whitelisted(ip)

def test_csv_storage_blacklist(csv_app_context):
    """Test CSV storage for blacklist operations."""
    ip = '10.10.10.1'
    reason = 'CSV test'
    
    # Test add and check
    add_ip_blacklist(ip, reason)
    assert is_ip_blacklisted(ip)
    
    # Test removal
    remove_ip_blacklist(ip)
    assert not is_ip_blacklisted(ip)


def test_csv_storage_blacklist_extended_request_info_column(csv_app_context):
    ip = "10.10.10.2"
    add_ip_blacklist(ip, "CSV extended", extended_request_info={"path": "/login"})

    csv_file = Path(csv_app_context.config["AIWAF_DATA_DIR"]) / "blacklist.csv"
    content = csv_file.read_text(encoding="utf-8")
    header = content.splitlines()[0]
    assert "extended_request_info" in header

def test_csv_storage_keywords(csv_app_context):
    """Test CSV storage for keywords."""
    keyword = 'csv_test_keyword'
    
    # Test add and check
    add_keyword(keyword)
    keywords = get_top_keywords()
    assert keyword in keywords

def test_csv_storage_geo_blocked_countries(csv_app_context):
    """Test CSV storage for geo blocked countries."""
    country = 'FR'

    add_geo_blocked_country(country)
    assert is_country_geo_blocked(country)
    assert country in get_geo_blocked_countries()

    remove_geo_blocked_country(country)
    assert not is_country_geo_blocked(country)
