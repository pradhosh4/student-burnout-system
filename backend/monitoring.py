import sqlite3
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression


DATABASE_NAME = "burnout_assessments.db"


def get_assessment_summary():

    connection = sqlite3.connect(DATABASE_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM assessments
    """)

    total = cursor.fetchone()[0]

    cursor.execute("""
        SELECT risk_level, COUNT(*)
        FROM assessments
        GROUP BY risk_level
    """)

    risk_counts = cursor.fetchall()

    connection.close()

    summary = {
        "total_assessments": total,
        "risk_counts": {
            "High": 0,
            "Medium": 0,
            "Low": 0
        }
    }

    for risk_level, count in risk_counts:
        summary["risk_counts"][risk_level] = count

    return summary
def get_new_assessment_count():

    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM assessments
    """)

    count = cursor.fetchone()[0]

    connection.close()

    return count
# Number of new assessments required before retraining
RETRAINING_THRESHOLD = 100


def should_retrain():

    new_assessment_count = get_new_assessment_count()

    return new_assessment_count >= RETRAINING_THRESHOLD
def retrain_model():

    if not should_retrain():
        return {
            "status": "not_required",
            "message": "Not enough new assessment data for retraining."
        }

    result = train_updated_model()

    return {
        "status": "completed",
        "message": "Model retraining completed successfully.",
        "details": result
    }
def train_updated_model():

    data_path = "data/student_mental_health_burnout.xls"

    # Load original dataset
    data = pd.read_csv(data_path)

    # Remove original burnout label
    data = data.drop(columns=["burnout_level"])

    # -----------------------------
    # Risk Feature Engineering
    # -----------------------------

    data["anxiety_risk"] = (
        data["anxiety_score"] - 1
    ) / (10 - 1)

    data["depression_risk"] = (
        data["depression_score"] - 1
    ) / (10 - 1)

    data["academic_pressure_risk"] = (
        data["academic_pressure_score"] - 1
    ) / (10 - 1)

    data["financial_stress_risk"] = (
        data["financial_stress_score"] - 1
    ) / (10 - 1)

    data["study_risk"] = (
        data["daily_study_hours"] - 1
    ) / (10 - 1)

    data["screen_risk"] = (
        data["screen_time_hours"] - 1
    ) / (12 - 1)

    data["sleep_risk"] = (
        9 - data["daily_sleep_hours"]
    ) / (9 - 4)

    data["activity_risk"] = (
        2 - data["physical_activity_hours"]
    ) / (2 - 0)

    data["social_support_risk"] = (
        10 - data["social_support_score"]
    ) / (10 - 1)

    # -----------------------------
    # Component Risk Scores
    # -----------------------------

    data["psychological_risk"] = (
        0.30 * data["anxiety_risk"] +
        0.30 * data["depression_risk"] +
        0.25 * data["academic_pressure_risk"] +
        0.15 * data["financial_stress_risk"]
    )

    data["workload_risk"] = (
        0.60 * data["study_risk"] +
        0.40 * data["screen_risk"]
    )

    data["lifestyle_risk"] = (
        0.40 * data["sleep_risk"] +
        0.30 * data["activity_risk"] +
        0.30 * data["social_support_risk"]
    )

    # -----------------------------
    # Composite Burnout Risk Score
    # -----------------------------

    data["burnout_risk_score"] = (
        0.45 * data["psychological_risk"] +
        0.30 * data["workload_risk"] +
        0.25 * data["lifestyle_risk"]
    )

    # -----------------------------
    # Create Burnout Risk Level
    # -----------------------------

    data["burnout_level"] = pd.qcut(
        data["burnout_risk_score"],
        q=3,
        labels=["Low", "Medium", "High"]
    )

    # -----------------------------
    # Prepare Features and Target
    # -----------------------------

    y = data["burnout_level"]

    X = data.drop(
        columns=[
            "burnout_level",
            "burnout_risk_score",
            "anxiety_risk",
            "depression_risk",
            "academic_pressure_risk",
            "financial_stress_risk",
            "study_risk",
            "screen_risk",
            "sleep_risk",
            "activity_risk",
            "social_support_risk",
            "psychological_risk",
            "workload_risk",
            "lifestyle_risk"
        ]
    )

    # Remove student ID
    X = X.drop(columns=["student_id"])

    # -----------------------------
    # Load Existing Preprocessor
    # -----------------------------

    preprocessor = joblib.load(
        "models/preprocessor.pkl"
    )

    X_processed = preprocessor.transform(X)

    # -----------------------------
    # Train Updated Logistic Model
    # -----------------------------

    model = LogisticRegression(
        C=100,
        max_iter=1000,
        random_state=42
    )

    model.fit(X_processed, y)

    # -----------------------------
    # Save Updated Model
    # -----------------------------

    joblib.dump(
        model,
        "models/burnout_logistic_model_updated.pkl"
    )

    return {
        "status": "success",
        "message": "Updated Logistic Regression model trained successfully."
    }
if __name__ == "__main__":
    print(train_updated_model())