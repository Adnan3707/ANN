import numpy as np
import pytest

from src.train import routing_metrics


def test_routing_metrics_only_scores_auto_routed_predictions() -> None:
    probabilities = np.array([
        [0.9, 0.05, 0.05],
        [0.6, 0.3, 0.1],
        [0.1, 0.8, 0.1],
    ])
    labels = np.array([0, 1, 0])

    result = routing_metrics(probabilities, labels, threshold=0.8)

    assert result["review_rate"] == pytest.approx(1 / 3)
    assert result["auto_route_rate"] == pytest.approx(2 / 3)
    assert result["auto_route_accuracy"] == pytest.approx(0.5)