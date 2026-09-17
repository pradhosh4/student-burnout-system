import sqlite3

DATABASE_NAME = "burnout_assessments.db"


def create_database():

    connection = sqlite3.connect(DATABASE_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessments (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            age INTEGER,
            gender TEXT,
            course TEXT,
            year TEXT,

            daily_study_hours REAL,
            daily_sleep_hours REAL,
            screen_time_hours REAL,

            stress_level TEXT,

            anxiety_score INTEGER,
            depression_score INTEGER,
            academic_pressure_score INTEGER,
            financial_stress_score INTEGER,
            social_support_score INTEGER,

            physical_activity_hours REAL,

            sleep_quality TEXT,

            attendance_percentage REAL,
            cgpa REAL,

            internet_quality TEXT,

            risk_level TEXT,

            high_probability REAL,
            medium_probability REAL,
            low_probability REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


def save_assessment(student_data, prediction, probabilities):

    connection = sqlite3.connect(DATABASE_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO assessments (

            age,
            gender,
            course,
            year,

            daily_study_hours,
            daily_sleep_hours,
            screen_time_hours,

            stress_level,

            anxiety_score,
            depression_score,
            academic_pressure_score,
            financial_stress_score,
            social_support_score,

            physical_activity_hours,

            sleep_quality,

            attendance_percentage,
            cgpa,

            internet_quality,

            risk_level,

            high_probability,
            medium_probability,
            low_probability

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

    """, (

        student_data["age"],
        student_data["gender"],
        student_data["course"],
        student_data["year"],

        student_data["daily_study_hours"],
        student_data["daily_sleep_hours"],
        student_data["screen_time_hours"],

        student_data["stress_level"],

        student_data["anxiety_score"],
        student_data["depression_score"],
        student_data["academic_pressure_score"],
        student_data["financial_stress_score"],
        student_data["social_support_score"],

        student_data["physical_activity_hours"],

        student_data["sleep_quality"],

        student_data["attendance_percentage"],
        student_data["cgpa"],

        student_data["internet_quality"],

        prediction,

        probabilities["High"],
        probabilities["Medium"],
        probabilities["Low"]
    ))

    connection.commit()
    connection.close()