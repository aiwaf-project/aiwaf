"""Shared rate counts for the native pipeline: Redis counts, aiwaf-rust decides."""
import json

import pytest

from aiwaf.core.rate_counter import RateCounter, RedisRateCounter, bounded_key

rust = pytest.importorskip("aiwaf_rust")
needs_rate_counter = pytest.mark.skipif(
    not getattr(getattr(rust, "NativeCSVPipeline", None), "supports", None)
    or not rust.NativeCSVPipeline.supports("rate_counter"), reason="aiwaf-rust without rate counters")


class FakeRedis:
    """Runs the regular limiter's sliding-window script the way Redis would."""

    def __init__(self):
        self.values = {}

    def eval(self, script, numkeys, key, now, window):
        now, window = float(now), float(window)
        current = [t for t in json.loads(self.values.get(key, "[]")) if now - t < window] + [now]
        self.values[key] = json.dumps(current)
        return self.values[key]


@pytest.fixture
def redis(monkeypatch):
    from aiwaf.core import cache_backend

    fake = FakeRedis()
    monkeypatch.setattr(cache_backend, "_redis_from_url", lambda url, **kwargs: fake)
    return fake


def test_redis_counter_counts_one_window_per_key_under_bounded_keys(redis):
    counter = RedisRateCounter("redis://cache:6379/0")
    assert [counter.count("ratelimit:1.2.3.4:/x", now, 10) for now in (100.0, 101.0, 111.0)] == [1, 2, 1]
    assert list(redis.values) == [bounded_key("aiwaf:native:rate:v1", "ratelimit:1.2.3.4:/x")]
    assert len(next(iter(redis.values))) == len("aiwaf:native:rate:v1:") + 64


@needs_rate_counter
def test_redis_backend_shares_one_limit_across_pipelines(tmp_path, redis):
    from aiwaf.core.csv_pipeline import CSVPipeline

    def worker(name):
        return CSVPipeline({"data_dir": str(tmp_path / name), "exempt_paths": [], "enabled": ["rate_limit"],
                            "manifest_path": str(tmp_path / "none.json"), "rate_max": 3, "rate_flood": 5,
                            "rate_backend": "redis", "rate_redis_url": "redis://cache:6379/0"})

    statuses = []
    for i, pipeline in enumerate([worker("a"), worker("b")] * 2):
        ctx, status, _ = pipeline.begin(ip="8.8.8.8", path="/x" * 500, method="GET", headers={}, query=b"",
                                        body=b"", exists=True, methods={"GET"})
        pipeline.finish(ctx, status or 200)
        statuses.append(status)
    assert statuses == [0, 0, 0, 429]


def test_the_interface_and_redis_settings_are_checked():
    with pytest.raises(NotImplementedError):
        RateCounter().count("k", 1.0, 1.0)
    with pytest.raises(ValueError, match="rate_redis_url"):
        RedisRateCounter("")
    with pytest.raises(ValueError, match="namespace"):
        RedisRateCounter("redis://cache", namespace="x" * 101)
