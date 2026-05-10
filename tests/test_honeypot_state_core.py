from aiwaf.core.honeypot import (
    store_honeypot_get_timestamp,
    load_honeypot_get_timestamp,
    clear_honeypot_get_timestamp,
)


def test_honeypot_state_helpers_use_consistent_key_and_ttl():
    calls = {}
    backing = {}

    def setter(key, value, ttl):
        calls['set'] = (key, value, ttl)
        backing[key] = value

    def getter(key):
        calls['get'] = key
        return backing.get(key)

    def deleter(key):
        calls['del'] = key
        backing.pop(key, None)

    store_honeypot_get_timestamp(setter, '203.0.113.10', 1234.5, ttl_seconds=111)
    assert calls['set'] == ('honeypot_get:203.0.113.10', 1234.5, 111)

    assert load_honeypot_get_timestamp(getter, '203.0.113.10') == 1234.5
    assert calls['get'] == 'honeypot_get:203.0.113.10'

    clear_honeypot_get_timestamp(deleter, '203.0.113.10')
    assert calls['del'] == 'honeypot_get:203.0.113.10'
    assert load_honeypot_get_timestamp(getter, '203.0.113.10') is None
