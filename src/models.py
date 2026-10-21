"""Keras model factories used by the FashionFinder experiments."""

from __future__ import annotations

import tensorflow as tf


IMAGE_SHAPE = (28, 28, 1)
NUM_CLASSES = 10


def compile_classifier(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
    """Compile a multi-class classifier with a consistent experiment contract."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def build_baseline(learning_rate: float = 1e-3) -> tf.keras.Model:
    """A dense-network baseline that ignores local image structure."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=IMAGE_SHAPE),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="dense_baseline",
    )
    return compile_classifier(model, learning_rate)


def build_cnn(learning_rate: float = 1e-3) -> tf.keras.Model:
    """A compact CNN suitable for CPU-friendly Fashion-MNIST experiments."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=IMAGE_SHAPE),
            tf.keras.layers.RandomTranslation(
                height_factor=0.1,
                width_factor=0.1,
                fill_mode="constant",
                fill_value=0.0,
            ),
            tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="cnn_candidate",
    )
    return compile_classifier(model, learning_rate)


def build_model(model_name: str, learning_rate: float = 1e-3) -> tf.keras.Model:
    """Return the requested model, failing clearly for invalid CLI choices."""
    factories = {"baseline": build_baseline, "cnn": build_cnn}
    try:
        return factories[model_name](learning_rate=learning_rate)
    except KeyError as error:
        raise ValueError(f"Unknown model '{model_name}'. Choose: {', '.join(factories)}") from error
