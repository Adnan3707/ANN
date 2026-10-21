import numpy as np

from src.models import build_model


def test_each_model_outputs_ten_class_probabilities() -> None:
    example = np.zeros((2, 28, 28, 1), dtype="float32")

    for model_name in ("baseline", "cnn"):
        probabilities = build_model(model_name)(example, training=False).numpy()
        assert probabilities.shape == (2, 10)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
