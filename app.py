# app.py
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi import Request
from fastapi.responses import HTMLResponse, FileResponse

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "template")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    html_path = os.path.join(TEMPLATE_DIR, "placement_predictor.html")
    return FileResponse(html_path, media_type="text/html")


model = joblib.load("placement_model.joblib")
columns = joblib.load("columns.joblib")


class StudentData(BaseModel):
    gender: str
    branch: str
    study_hours: float
    attendance: float
    sleep_hours: float
    internet_usage: float
    assignments_completed: int
    previous_score: float
    extracurricular: str
    exam_score: float


@app.post("/predict")
def predict(data: StudentData):
    gender = 1 if data.gender.lower() == "male" else 0
    extracurricular = 1 if data.extracurricular.lower() == "yes" else 0

    branch = data.branch.lower()

    branch_CIVIL = 1 if branch == "civil" else 0
    branch_COMMERCE = 1 if branch == "commerce" else 0
    branch_CSE = 1 if branch == "cse" else 0
    branch_ECE = 1 if branch == "ece" else 0
    branch_IT = 1 if branch == "it" else 0
    branch_MECHANICAL = 1 if branch == "mechanical" else 0

    row = {
        "gender": gender,
        "branch": branch,
        "study_hours": data.study_hours,
        "attendance": data.attendance,
        "sleep_hours": data.sleep_hours,
        "internet_usage": data.internet_usage,
        "assignments_completed": data.assignments_completed,
        "previous_score": data.previous_score,
        "extracurricular": extracurricular,
        "exam_score": data.exam_score,
        # If your model actually expects the one-hot columns, add them too:
        "branch_CIVIL": branch_CIVIL,
        "branch_COMMERCE": branch_COMMERCE,
        "branch_CSE": branch_CSE,
        "branch_ECE": branch_ECE,
        "branch_IT": branch_IT,
        "branch_MECHANICAL": branch_MECHANICAL,
    }

    df_row = pd.DataFrame([row]).reindex(columns=columns, fill_value=0)
    pred = model.predict(df_row)[0]

    return {"prediction": "Placed" if pred == 1 else "Not Placed"}
