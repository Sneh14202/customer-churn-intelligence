import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss
)


def load_model(model_path):
    """
    Load a saved machine-learning model.

    Parameters
    ----------
    model_path : str
        Path to the saved model.

    Returns
    -------
    model
        Loaded model.
    """

    return joblib.load(model_path)


def generate_predictions(model, X):
    """
    Generate churn probabilities and predictions.

    Parameters
    ----------
    model : trained model
        Trained classification model.

    X : pandas.DataFrame
        Model features.

    Returns
    -------
    probabilities : numpy.ndarray
        Predicted probability of churn.

    predictions : numpy.ndarray
        Binary predictions using threshold 0.50.
    """

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    return probabilities, predictions


def calculate_metrics(y_true, probabilities, threshold=0.50):
    """
    Calculate classification and probability metrics.

    Parameters
    ----------
    y_true : array-like
        Actual target values.

    probabilities : array-like
        Predicted churn probabilities.

    threshold : float
        Classification threshold.

    Returns
    -------
    dict
        Model performance metrics.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "ROC_AUC": roc_auc_score(
            y_true,
            probabilities
        ),

        "PR_AUC": average_precision_score(
            y_true,
            probabilities
        ),

        "Brier_Score": brier_score_loss(
            y_true,
            probabilities
        ),

        "Log_Loss": log_loss(
            y_true,
            probabilities
        )
    }

    return metrics


def score_customers(model, X):
    """
    Generate churn probabilities for customers.

    Parameters
    ----------
    model : trained model
        Trained calibrated churn model.

    X : pandas.DataFrame
        Model features.

    Returns
    -------
    pandas.Series
        Churn probabilities.
    """

    probabilities = model.predict_proba(X)[:, 1]

    return pd.Series(
        probabilities,
        index=X.index,
        name="ChurnProbability"
    )