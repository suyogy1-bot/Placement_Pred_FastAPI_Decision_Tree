import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "template"))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("placement_predictor.html", {"request": request})

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
    gender = 1 if data.gender.lower() == "Male" else 0
    extracurricular = 1 if data.extracurricular.lower() == 'Yes' else 0

    branch = data.branch.lower()

    branch_CIVIL = 1 if branch == "CIVIL" else 0
    branch_COMMERCE = 1 if branch == "COMMERCE" else 0
    branch_CSE = 1 if branch == "CSE" else 0
    branch_ECE = 1 if branch == "ECE" else 0
    branch_IT = 1 if branch == "IT" else 0
    branch_MECHANICAL = 1 if branch == "MECHANICAL" else 0

    row = {
        "gender" : gender,
        "branch" : branch,
        "study_hours" : data.study_hours,
        "attendance" : data.attendance,
        "sleep_hours" : data.sleep_hours,
        "internet_usage" : data.internet_usage,
        "assignments_completed" : data.assignments_completed,
        "previous_score" : data.previous_score,
        "extracurricular" : extracurricular,
        "exam_score" : data.exam_score
    }

    df_row = pd.DataFrame([row]).reindex(columns=columns, fill_value=0)
    pred = model.predict(df_row)[0]

    return {"prediction": "Placed" if pred == 1 else "Not Placed"}


