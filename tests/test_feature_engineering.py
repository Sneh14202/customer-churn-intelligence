import pandas as pd

from src.feature_engineering import (
    clean_raw_data,
    create_modeling_data
)


def test_total_charges_is_numeric():
    """
    TotalCharges should be converted to a numeric column.
    """

    df = pd.DataFrame({
        "customerID": ["001", "002"],
        "TotalCharges": ["100.50", "200.75"],
        "Churn": ["No", "Yes"]
    })

    cleaned = clean_raw_data(df)

    assert pd.api.types.is_numeric_dtype(
        cleaned["TotalCharges"]
    )


def test_missing_total_charges_are_filled():
    """
    Missing TotalCharges values should be handled.
    """

    df = pd.DataFrame({
        "customerID": ["001", "002", "003"],
        "TotalCharges": ["100.50", "", "200.75"],
        "Churn": ["No", "Yes", "No"]
    })

    cleaned = clean_raw_data(df)

    assert cleaned["TotalCharges"].isna().sum() == 0


def test_customer_id_is_removed_from_features():
    """
    customerID should not be used as a model feature.
    """

    df = pd.DataFrame({
        "customerID": ["001", "002"],
        "gender": ["Male", "Female"],
        "TotalCharges": ["100", "200"],
        "Churn": ["No", "Yes"]
    })

    X, y = create_modeling_data(df)

    assert "customerID" not in X.columns


def test_churn_is_binary():
    """
    Churn should be converted to 0/1.
    """

    df = pd.DataFrame({
        "customerID": ["001", "002"],
        "gender": ["Male", "Female"],
        "TotalCharges": ["100", "200"],
        "Churn": ["No", "Yes"]
    })

    X, y = create_modeling_data(df)

    assert set(y.unique()).issubset({0, 1})


def test_model_features_are_numeric():
    """
    All model features should be numeric after encoding.
    """

    df = pd.DataFrame({
        "customerID": ["001", "002"],
        "gender": ["Male", "Female"],
        "Contract": [
            "Month-to-month",
            "Two year"
        ],
        "TotalCharges": ["100", "200"],
        "Churn": ["No", "Yes"]
    })

    X, y = create_modeling_data(df)

    assert all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in X.dtypes
    )