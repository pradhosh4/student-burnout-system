from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from backend.database import create_database
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
app = FastAPI(
    title="Student Burnout Prediction API",
    description="AI-based Student Burnout Prediction and Early Intervention System",
    version="1.0.0"
)
create_database()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/burnout_logistic_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")


class StudentData(BaseModel):
    age: int
    gender: str
    course: str
    year: str
    daily_study_hours: float
    daily_sleep_hours: float
    screen_time_hours: float
    stress_level: str
    anxiety_score: int
    depression_score: int
    academic_pressure_score: int
    financial_stress_score: int
    social_support_score: int
    physical_activity_hours: float
    sleep_quality: str
    attendance_percentage: float
    cgpa: float
    internet_quality: str


@app.get("/")
def home():
    return FileResponse("frontend/index.html")
@app.post("/predict")
def predict(student: StudentData):
    student_dict = student.model_dump()

    import pandas as pd

    student_df = pd.DataFrame([student_dict])

    processed_data = preprocessor.transform(student_df)

    prediction = model.predict(processed_data)[0]

    probabilities = model.predict_proba(processed_data)[0]

    probability_dict = {
        class_name: round(float(probability) * 100, 2)
        for class_name, probability
        in zip(model.classes_, probabilities)
    }
    from backend.database import save_assessment

    save_assessment(
    student_dict,
    prediction,
    probability_dict
)
    return {
        "risk_level": prediction,
        "probabilities": probability_dict
    }
@app.get("/monitoring")
def monitoring():

    from backend.monitoring import get_assessment_summary

    return get_assessment_summary()
@app.get("/retraining-status")
def retraining_status():

    from backend.monitoring import (
        get_new_assessment_count,
        should_retrain,
        retrain_model,
        RETRAINING_THRESHOLD
    )

    count = get_new_assessment_count()

    if should_retrain():
        result = retrain_model()

        return {
            "assessment_count": count,
            "retraining_threshold": RETRAINING_THRESHOLD,
            "retraining_required": True,
            "retraining_status": result["status"],
            "message": result["message"]
        }

    return {
        "assessment_count": count,
        "retraining_threshold": RETRAINING_THRESHOLD,
        "retraining_required": False,
        "retraining_status": "not_required",
        "message": "Not enough new assessment data for retraining."
    }

app.mount(
    "/",
    StaticFiles(directory="frontend", html=True),
    name="frontend"
)