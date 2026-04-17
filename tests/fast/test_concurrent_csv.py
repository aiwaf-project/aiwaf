from concurrent.futures import ThreadPoolExecutor

from aiwaf.fast.storage import get_blacklist_store, initialize_storage


def test_concurrent_blacklist_writes_memory_backend():
    initialize_storage(backend="memory")
    store = get_blacklist_store()
    ips = [f"203.0.113.{idx}" for idx in range(1, 21)]

    def _add(ip):
        return store.block_ip(ip, "concurrency test")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(_add, ips))

    # block_ip mutates store and returns None; validate outcomes via state.
    assert all(result is None for result in results)
    assert set(ips).issubset(set(store.get_blocked_ips()))
