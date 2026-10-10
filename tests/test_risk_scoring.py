import pandas as pd

from src.risk_scoring import (
    assign_risk_segment,
    assign_risk_priority,
    assign_retention_action,
    create_risk_scores,
    create_risk_summary
)


def test_low_risk_segment():

    assert assign_risk_segment(0.05) == "Low"


def test_medium_risk_segment():

    assert assign_risk_segment(0.20) == "Medium"


def test_high_risk_segment():

    assert assign_risk_segment(0.40) == "High"


def test_critical_risk_segment():

    assert assign_risk_segment(0.75) == "Critical"


def test_risk_threshold_boundaries():

    assert assign_risk_segment(0.10) == "Medium"

    assert assign_risk_segment(0.30) == "High"

    assert assign_risk_segment(0.50) == "Critical"


def test_risk_priority():

    assert assign_risk_priority("Low") == 1

    assert assign_risk_priority("Medium") == 2

    assert assign_risk_priority("High") == 3

    assert assign_risk_priority("Critical") == 4


def test_retention_actions():

    assert (
        assign_retention_action("Low")
        == "Normal engagement and monitoring"
    )

    assert (
        assign_retention_action("Medium")
        == "Targeted engagement and monitoring"
    )

    assert (
        assign_retention_action("High")
        == "Proactive retention campaign"
    )

    assert (
        assign_retention_action("Critical")
        == "Immediate personalized intervention"
    )


def test_create_risk_scores():

    df = pd.DataFrame({
        "customerID": [
            "001",
            "002",
            "003",
            "004"
        ],
        "ChurnProbability": [
            0.05,
            0.20,
            0.40,
            0.75
        ]
    })

    result = create_risk_scores(df)

    assert list(result["RiskSegment"]) == [
        "Low",
        "Medium",
        "High",
        "Critical"
    ]

    assert list(result["RiskPriority"]) == [
        1,
        2,
        3,
        4
    ]

    assert "RetentionAction" in result.columns


def test_create_risk_summary():

    df = pd.DataFrame({
        "ChurnProbability": [
            0.05,
            0.20,
            0.40,
            0.75
        ],
        "RiskSegment": [
            "Low",
            "Medium",
            "High",
            "Critical"
        ]
    })

    summary = create_risk_summary(df)

    assert len(summary) == 4

    assert "CustomerCount" in summary.columns

    assert "AverageProbability" in summary.columns

    assert "CustomerPercentage" in summary.columns

    assert summary["CustomerCount"].sum() == 4