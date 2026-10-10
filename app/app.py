import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# 1. PROJECT PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.feature_engineering import create_modeling_data
from src.model_utils import load_model
from src.risk_scoring import (
    assign_risk_segment,
    assign_risk_priority,
    assign_retention_action
)


# ============================================================
# 2. FILE PATHS
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "calibrated_gradient_boosting.joblib"
)

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

RISK_DATA_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "customer_risk_scores_with_explanations.csv"
)

RISK_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "risk_segment_summary.csv"
)

KPI_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "executive_churn_kpis.csv"
)


# ============================================================
# 3. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_churn_model():
    return load_model(MODEL_PATH)


@st.cache_data
def load_customer_data():
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_data
def load_risk_data():
    return pd.read_csv(RISK_DATA_PATH)


@st.cache_data
def load_risk_summary():
    return pd.read_csv(RISK_SUMMARY_PATH)


@st.cache_data
def load_kpis():
    return pd.read_csv(KPI_PATH)


model = load_churn_model()
customer_data = load_customer_data()
risk_data = load_risk_data()
risk_summary = load_risk_summary()
kpis = load_kpis()


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def create_customer_dataframe(
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges
):
    """
    Create a one-row dataframe using the same
    feature names as the original dataset.
    """

    customer = pd.DataFrame([{
        "customerID": "APP_CUSTOMER",
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": "No"
    }])

    return customer


def prepare_customer_features(customer_df):
    """
    Prepare customer input using the same feature engineering
    approach used during model development.
    """

    X, _ = create_modeling_data(customer_df)

    return X


def align_features(X, expected_features):
    """
    Make sure the input dataframe has exactly the same
    columns as the model expects.
    """

    X = X.copy()

    for column in expected_features:
        if column not in X.columns:
            X[column] = 0

    X = X[expected_features]

    return X


def get_probability_color(probability):
    """
    Return a Streamlit status based on churn probability.
    """

    if probability < 0.10:
        return "Low"

    elif probability < 0.30:
        return "Medium"

    elif probability < 0.50:
        return "High"

    else:
        return "Critical"


# ============================================================
# 6. SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page:",
    [
        "Executive Dashboard",
        "Customer Risk Scoring",
        "Risk Segments"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Customer Churn Prediction & Retention Intelligence Platform**

    Built using:

    - Python
    - Pandas
    - Scikit-learn
    - Gradient Boosting
    - SHAP
    - Streamlit
    """
)


# ============================================================
# 7. EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":

    st.title(
        "📊 Customer Churn Prediction & Retention Intelligence"
    )

    st.markdown(
        """
        ### Executive Dashboard

        This dashboard summarizes customer churn risk,
        model-driven customer prioritization, and retention intelligence.
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # KPI SECTION
    # --------------------------------------------------------

    total_customers = len(customer_data)

    total_churned = int(
        customer_data["Churn"]
        .eq("Yes")
        .sum()
    )

    churn_rate = (
        total_churned
        / total_customers
        * 100
    )

    high_risk = int(
        (risk_data["RiskSegment"] == "High").sum()
    )

    critical_risk = int(
        (risk_data["RiskSegment"] == "Critical").sum()
    )

    priority_customers = (
        high_risk
        + critical_risk
    )

    priority_percentage = (
        priority_customers
        / total_customers
        * 100
    )

    average_probability = (
        risk_data["ChurnProbability"].mean()
        * 100
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Churned Customers",
        f"{total_churned:,}"
    )

    col3.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

    col4.metric(
        "High + Critical",
        f"{priority_customers:,}"
    )

    col5.metric(
        "Priority %",
        f"{priority_percentage:.2f}%"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Customer Risk Distribution")

    risk_chart_data = (
        risk_data["RiskSegment"]
        .value_counts()
        .reindex(
            ["Low", "Medium", "High", "Critical"]
        )
        .fillna(0)
    )

    st.bar_chart(risk_chart_data)

    st.markdown("---")

    # --------------------------------------------------------
    # BUSINESS INTERPRETATION
    # --------------------------------------------------------

    st.subheader("Business Interpretation")

    st.write(
        f"""
        The dataset contains **{total_customers:,} customers**,
        of which **{total_churned:,} have churned**.

        The overall historical churn rate is approximately
        **{churn_rate:.2f}%**.

        The model identifies **{priority_customers:,} customers**
        as High or Critical risk, representing approximately
        **{priority_percentage:.2f}% of the customer base**.

        The average predicted churn probability is
        approximately **{average_probability:.2f}%**.
        """
    )


# ============================================================
# 8. CUSTOMER RISK SCORING
# ============================================================

elif page == "Customer Risk Scoring":

    st.title("🎯 Customer Risk Scoring")

    st.markdown(
        """
        Enter customer information below to generate an individual
        churn-risk assessment.
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # CUSTOMER PROFILE
    # --------------------------------------------------------

    st.subheader("Customer Profile")

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

    with col2:

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No phone service",
                "No",
                "Yes"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col3:

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    # --------------------------------------------------------
    # CONTRACT AND BILLING
    # --------------------------------------------------------

    st.subheader("Contract & Billing")

    col1, col2, col3 = st.columns(3)

    with col1:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    with col2:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col3:

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=monthly_charges * max(tenure, 1),
            step=10.0
        )

    st.markdown("---")

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predict_button = st.button(
        "🔮 Predict Churn Risk",
        type="primary"
    )

    if predict_button:

        customer_df = create_customer_dataframe(
            gender=gender,
            senior_citizen=senior_citizen,
            partner=partner,
            dependents=dependents,
            tenure=tenure,
            phone_service=phone_service,
            multiple_lines=multiple_lines,
            internet_service=internet_service,
            online_security=online_security,
            online_backup=online_backup,
            device_protection=device_protection,
            tech_support=tech_support,
            streaming_tv=streaming_tv,
            streaming_movies=streaming_movies,
            contract=contract,
            paperless_billing=paperless_billing,
            payment_method=payment_method,
            monthly_charges=monthly_charges,
            total_charges=total_charges
        )

        X_customer = prepare_customer_features(
            customer_df
        )

        # Get the feature names expected by the trained model
        expected_features = (
            getattr(
                model,
                "feature_names_in_",
                None
            )
        )

        if expected_features is not None:

            X_customer = align_features(
                X_customer,
                expected_features
            )

        probability = (
            model
            .predict_proba(X_customer)[0][1]
        )

        risk_segment = assign_risk_segment(
            probability
        )

        risk_priority = assign_risk_priority(
            risk_segment
        )

        retention_action = assign_retention_action(
            risk_segment
        )

        st.markdown("---")

        st.subheader("Prediction Result")

        result_col1, result_col2, result_col3, result_col4 = (
            st.columns(4)
        )

        result_col1.metric(
            "Churn Probability",
            f"{probability * 100:.1f}%"
        )

        result_col2.metric(
            "Risk Segment",
            risk_segment
        )

        result_col3.metric(
            "Risk Priority",
            str(risk_priority)
        )

        result_col4.metric(
            "Model Decision",
            "High Risk"
            if probability >= 0.50
            else "Lower Risk"
        )

        st.progress(
            float(probability)
        )

        st.markdown("---")

        if risk_segment == "Low":

            st.success(
                f"""
                **Low Risk**

                Recommended action:

                {retention_action}
                """
            )

        elif risk_segment == "Medium":

            st.info(
                f"""
                **Medium Risk**

                Recommended action:

                {retention_action}
                """
            )

        elif risk_segment == "High":

            st.warning(
                f"""
                **High Risk**

                Recommended action:

                {retention_action}
                """
            )

        else:

            st.error(
                f"""
                **Critical Risk**

                Recommended action:

                {retention_action}
                """
            )


# ============================================================
# 9. RISK SEGMENTS
# ============================================================

elif page == "Risk Segments":

    st.title("🚦 Customer Risk Segments")

    st.markdown(
        """
        Customer risk segments are based on calibrated churn probability.
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # RISK TABLE
    # --------------------------------------------------------

    display_columns = [
        column
        for column in [
            "RiskSegment",
            "CustomerCount",
            "CustomerPercentage",
            "AverageProbability",
            "ActualChurnRate"
        ]
        if column in risk_summary.columns
    ]

    if display_columns:

        st.dataframe(
            risk_summary[display_columns],
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # RISK CHART
    # --------------------------------------------------------

    st.subheader("Customers by Risk Segment")

    chart_data = (
        risk_data["RiskSegment"]
        .value_counts()
        .reindex(
            ["Low", "Medium", "High", "Critical"]
        )
        .fillna(0)
    )

    st.bar_chart(chart_data)

    # --------------------------------------------------------
    # HIGH PRIORITY CUSTOMERS
    # --------------------------------------------------------

    st.subheader("Highest-Risk Customers")

    high_priority = (
        risk_data[
            risk_data["RiskSegment"].isin(
                ["High", "Critical"]
            )
        ]
        .sort_values(
            "ChurnProbability",
            ascending=False
        )
    )

    display_customer_columns = [
        column
        for column in [
            "customerID",
            "ChurnProbability",
            "RiskSegment",
            "RiskDrivers",
            "RetentionRecommendations"
        ]
        if column in high_priority.columns
    ]

    st.dataframe(
        high_priority[
            display_customer_columns
        ].head(25),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Showing the 25 highest-risk High/Critical customers."
    )


# ============================================================
# 10. FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Customer Churn Prediction & Retention Intelligence Platform"
)

st.sidebar.caption(
    "Portfolio Project | Python | Scikit-learn | SHAP | Streamlit"
)