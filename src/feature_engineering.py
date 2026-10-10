import pandas as pd


def clean_raw_data(df):
    """
    Clean the raw Telco customer churn dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw customer churn dataframe.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe.
    """

    df = df.copy()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing TotalCharges with median
    if df["TotalCharges"].isna().sum() > 0:
        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["TotalCharges"].median()
        )

    return df


def create_modeling_data(df):
    """
    Convert cleaned customer data into machine-learning features.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned customer dataframe.

    Returns
    -------
    X : pandas.DataFrame
        Model features.
    y : pandas.Series
        Binary churn target.
    """

    df = clean_raw_data(df)

    # Convert target to binary
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # Remove customer identifier
    df = df.drop(columns=["customerID"])

    # Separate target
    y = df["Churn"]

    X = df.drop(columns=["Churn"])

    # Identify categorical columns
    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # One-hot encode categorical variables
    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )

    return X, y