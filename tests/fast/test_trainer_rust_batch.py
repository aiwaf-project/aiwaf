from aiwaf.core.training import extract_rust_features_parallel


def test_rust_batch_parallel_extraction_passthrough():
    records = [{"x": 1}, {"x": 2}]

    def _extract(chunk, _keywords):
        return [[item["x"]] for item in chunk]

    features = extract_rust_features_parallel(records, [], chunk_size=1, max_workers=2, extract_fn=_extract)
    assert features == [[1], [2]]

