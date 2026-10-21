"""Dataset loading and deterministic preprocessing for Fashion-MNIST."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import tensorflow as tf


CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

FASHION_MNIST_BASE_URL = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
)
FASHION_MNIST_FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}


def load_idx_gzip(file_path: str | Path) -> np.ndarray:
    """Read a gzipped IDX array used by the original Fashion-MNIST release."""
    with gzip.open(file_path, "rb") as dataset_file:
        zero, zero_again, data_type, dimensions = struct.unpack(
            ">BBBB", dataset_file.read(4)
        )
        if (zero, zero_again, data_type) != (0, 0, 8):
            raise ValueError(f"{file_path} is not an unsigned-byte IDX file")
        shape = struct.unpack(f">{'I' * dimensions}", dataset_file.read(4 * dimensions))
        values = np.frombuffer(dataset_file.read(), dtype=np.uint8)
    return values.reshape(shape)


def download_fashion_mnist(data_path: Path) -> dict[str, np.ndarray]:
    """Download Fashion-MNIST from the project source and read its IDX files."""
    arrays: dict[str, np.ndarray] = {}
    for split_name, file_name in FASHION_MNIST_FILES.items():
        local_file = tf.keras.utils.get_file(
            fname=file_name,
            origin=f"{FASHION_MNIST_BASE_URL}/{file_name}",
            cache_dir=str(data_path.resolve()),
            cache_subdir="",
        )
        arrays[split_name] = load_idx_gzip(local_file)
    return arrays


def load_fashion_mnist(
    data_dir: str | Path = "data",
    validation_size: int = 5_000,
    seed: int = 42,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load, normalize, and deterministically split Fashion-MNIST.

    Returns `(train, validation, test)`, with image values in `[0, 1]` and
    shapes `(examples, 28, 28, 1)`. The official test split is never used to
    choose a model.
    """
    if not 0 < validation_size < 60_000:
        raise ValueError("validation_size must be between 1 and 59,999")

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    # Keras no longer supports a project-local path for Fashion-MNIST, so load
    # the original public IDX files into this repository's data directory.
    arrays = download_fashion_mnist(data_path)
    x_train = arrays["x_train"]
    y_train = arrays["y_train"]
    x_test = arrays["x_test"]
    y_test = arrays["y_test"]

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Store a single shuffled split so each model is compared against identical data.
    generator = np.random.default_rng(seed)
    indices = generator.permutation(len(x_train))
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]

    return (
        (x_train[training_indices], y_train[training_indices]),
        (x_train[validation_indices], y_train[validation_indices]),
        (x_test, y_test),
    )


def make_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    training: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Build a performant and repeatable tf.data input pipeline."""
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(images), 10_000), seed=seed, reshuffle_each_iteration=True
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
