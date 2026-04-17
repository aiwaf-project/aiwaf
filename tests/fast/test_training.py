from aiwaf.core.training import iter_batches


def test_training_batch_iterator():
    batches = list(iter_batches([1, 2, 3, 4, 5], batch_size=2))
    assert batches == [[1, 2], [3, 4], [5]]

