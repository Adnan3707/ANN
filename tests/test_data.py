import numpy as np

from src.data import make_dataset


def test_make_dataset_batches_all_examples() -> None:
    images = np.zeros((3, 28, 28, 1), dtype="float32")
    labels = np.array([0, 1, 2])

    batches = list(make_dataset(images, labels, batch_size=2))

    assert [len(batch_labels) for _, batch_labels in batches] == [2, 1]

