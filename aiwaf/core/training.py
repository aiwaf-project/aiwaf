"""
Shared training helpers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor


def iter_batches(items, batch_size: int):
    batch_size = max(1, int(batch_size))
    for idx in range(0, len(items), batch_size):
        yield items[idx:idx + batch_size]


def extract_rust_features_parallel(records, static_keywords, chunk_size, max_workers, extract_fn):
    """Parallel feature extraction for Rust backends without stateful batching."""
    if not records:
        return []

    chunk_size = max(1, int(chunk_size))
    max_workers = max(1, int(max_workers))
    if max_workers == 1 or len(records) <= chunk_size:
        return extract_fn(records, static_keywords)

    chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]
    features = []

    def _extract_chunk(chunk):
        return extract_fn(chunk, static_keywords)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunk_results = list(executor.map(_extract_chunk, chunks))

    for result in chunk_results:
        if result is None:
            return None
        features.extend(result)
    return features
