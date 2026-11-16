"""Native rate limits shared through a real Redis."""
import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get('AIWAF_TEST_REDIS_URL'),
    reason='Set AIWAF_TEST_REDIS_URL to a dedicated test Redis instance')


def test_two_pipelines_share_one_redis_limit(tmp_path):
    rust = pytest.importorskip('aiwaf_rust')
    if not rust.NativeCSVPipeline.supports('rate_counter'):
        pytest.skip('aiwaf-rust without rate counters')
    from aiwaf.core.csv_pipeline import CSVPipeline

    def worker(name):
        return CSVPipeline({'data_dir': str(tmp_path / name), 'exempt_paths': [], 'enabled': ['rate_limit'],
                            'manifest_path': str(tmp_path / 'none.json'), 'rate_max': 2, 'rate_flood': 10,
                            'rate_backend': 'redis', 'rate_redis_url': os.environ['AIWAF_TEST_REDIS_URL'],
                            'rate_cache_namespace': f'aiwaf:test:{uuid.uuid4().hex}'})

    workers = [worker('a'), worker('b')]
    statuses = []
    for i in range(3):
        ctx, status, _ = workers[i % 2].begin(ip='198.51.100.9', path='/api', method='GET', headers={},
                                              query=b'', body=b'', exists=True, methods={'GET'})
        workers[i % 2].finish(ctx, status or 200)
        statuses.append(status)
    assert statuses == [0, 0, 429]
