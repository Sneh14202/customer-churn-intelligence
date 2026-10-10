import pandas as pd


def assign_risk_segment(probability):
    """
    Assign a customer to a churn-risk segment.

    Risk thresholds:

    Low       < 0.10
    Medium    0.10 - <0.30
    High      0.30 - <0.50
    Critical  >= 0.50
    """

    if probability < 0.10:
        return "Low"

    elif probability < 0.30:
        return "Medium"

    elif probability < 0.50:
        return "High"

    else:
        return "Critical"


def assign_risk_priority(segment):
    """
    Assign numerical priority to a risk segment.
    """

    priority_map = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Critical": 4
    }

    return priority_map.get(segment, 0)


def assign_retention_action(segment):
    """
    Assign a business action based on risk segment.
    """

    action_map = {
        "Low": "Normal engagement and monitoring",

        "Medium": "Targeted engagement and monitoring",

        "High": "Proactive retention campaign",

        "Critical": "Immediate personalized intervention"
    }

    return action_map.get(
        segment,
        "Review customer"
    )


def create_risk_scores(df, probability_column="ChurnProbability"):
    """
    Create risk segments and priorities for customers.

    Parameters
    ----------
    df : pandas.DataFrame
        Customer dataframe containing churn probabilities.

    probability_column : str
        Name of probability column.

    Returns
    -------
    pandas.DataFrame
        Customer dataframe with risk information.
    """

    result = df.copy()

    result["RiskSegment"] = (
        result[probability_column]
        .apply(assign_risk_segment)
    )

    result["RiskPriority"] = (
        result["RiskSegment"]
        .apply(assign_risk_priority)
    )

    result["RetentionAction"] = (
        result["RiskSegment"]
        .apply(assign_retention_action)
    )

    return result


def create_risk_summary(df):
    """
    Create summary statistics by risk segment.
    """

    summary = (
        df.groupby("RiskSegment")
        .agg(
            CustomerCount=("RiskSegment", "size"),
            AverageProbability=(
                "ChurnProbability",
                "mean"
            )
        )
        .reset_index()
    )

    segment_order = [
        "Low",
        "Medium",
        "High",
        "Critical"
    ]

    summary["RiskSegment"] = pd.Categorical(
        summary["RiskSegment"],
        categories=segment_order,
        ordered=True
    )

    summary = (
        summary
        .sort_values("RiskSegment")
        .reset_index(drop=True)
    )

    summary["CustomerPercentage"] = (
        summary["CustomerCount"]
        / summary["CustomerCount"].sum()
        * 100
    )

    return summary