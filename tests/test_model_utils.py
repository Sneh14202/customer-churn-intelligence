import numpy as np

from src.model_utils import (
    generate_predictions,
    calculate_metrics,
    score_customers
)


class MockModel:

    def predict_proba(self, X):
        """
        Return predictable probabilities for testing.
        """

        probabilities = np.array([
            [0.90, 0.10],
            [0.30, 0.70],
            [0.80, 0.20],
            [0.20, 0.80]
        ])

        return probabilities


def test_generate_predictions():

    model = MockModel()

    X = np.zeros((4, 2))

    probabilities, predictions = generate_predictions(
        model,
        X
    )

    assert len(probabilities) == 4

    assert len(predictions) == 4

    assert list(predictions) == [0, 1, 0, 1]


def test_calculate_metrics():

    y_true = np.array([
        0,
        1,
        0,
        1
    ])

    probabilities = np.array([
        0.10,
        0.70,
        0.20,
        0.80
    ])

    metrics = calculate_metrics(
        y_true,
        probabilities
    )

    assert "Accuracy" in metrics

    assert "Precision" in metrics

    assert "Recall" in metrics

    assert "F1" in metrics

    assert "ROC_AUC" in metrics

    assert "PR_AUC" in metrics

    assert "Brier_Score" in metrics

    assert "Log_Loss" in metrics


def test_perfect_predictions_have_high_auc():

    y_true = np.array([
        0,
        0,
        1,
        1
    ])

    probabilities = np.array([
        0.05,
        0.10,
        0.90,
        0.95
    ])

    metrics = calculate_metrics(
        y_true,
        probabilities
    )

    assert metrics["ROC_AUC"] == 1.0


def test_score_customers():

    model = MockModel()

    X = np.zeros((4, 2))

    scores = score_customers(
        model,
        X
    )

    assert len(scores) == 4

    assert scores.name == "ChurnProbability"

    assert scores.iloc[0] == 0.10