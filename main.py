"""
Employee Attrition Prediction API

A FastAPI service that wraps the trained attrition model (Task 2c) so predictions can be
requested programmatically by the n8n agentic workflow (Task 2e).

The joblib artifact is a *complete* scikit-learn Pipeline (preprocessing + Logistic Regression),
so the API applies exactly the same transformations used at training time — no training/serving
skew. Run locally with:

    uvicorn main:app --reload

Interactive docs are served at /docs (Swagger UI) and /redoc.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# load artifacts
HERE = Path(__file__).parent
PIPELINE = joblib.load(HERE / "attrition_pipeline.joblib")
META = json.loads((HERE / "model_metadata.json").read_text(encoding="utf-8"))

FEATURE_ORDER: list[str] = META["raw_feature_order"]
THRESHOLD: float = META["decision_threshold"]
TIER_CUTS: dict = META["tier_cuts"]
MEDIANS: dict = META["feature_medians"]

app = FastAPI(
    title="Employee Attrition Prediction API",
    description=(
        "Predicts the probability that an employee will leave the organisation, returns a risk "
        "tier, and lists the individual risk factors that apply — designed to be consumed by an "
        "n8n AI-agent workflow. Model: %s (AIT403 Task 2c)." % META["model_name"]
    ),
    version="1.0.0",
)


# schemas
class Employee(BaseModel):
    """One employee's attributes. Field names match the IBM HR dataset columns.

    `employee_name` / `employee_id` are optional identifiers echoed back in the response
    (useful for the alert message); they are not used by the model.
    """

    employee_name: Optional[str] = Field(None, description="Optional display name for alerts")
    employee_id: Optional[int] = Field(None, description="Optional employee identifier")

    Age: int
    BusinessTravel: str = Field(..., description="Non-Travel | Travel_Rarely | Travel_Frequently")
    DailyRate: int
    Department: str = Field(..., description="Sales | Research & Development | Human Resources")
    DistanceFromHome: int
    Education: int = Field(..., ge=1, le=5)
    EducationField: str
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4)
    Gender: str = Field(..., description="Male | Female")
    HourlyRate: int
    JobInvolvement: int = Field(..., ge=1, le=4)
    JobLevel: int = Field(..., ge=1, le=5)
    JobRole: str
    JobSatisfaction: int = Field(..., ge=1, le=4)
    MaritalStatus: str = Field(..., description="Single | Married | Divorced")
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: str = Field(..., description="Yes | No")
    PercentSalaryHike: int
    PerformanceRating: int = Field(..., ge=1, le=4)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4)
    StockOptionLevel: int = Field(..., ge=0, le=3)
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int = Field(..., ge=1, le=4)
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "employee_name": "Jordan Lee",
                "employee_id": 1042,
                "Age": 29, "BusinessTravel": "Travel_Frequently", "DailyRate": 800,
                "Department": "Sales", "DistanceFromHome": 15, "Education": 3,
                "EducationField": "Marketing", "EnvironmentSatisfaction": 1, "Gender": "Male",
                "HourlyRate": 60, "JobInvolvement": 2, "JobLevel": 1,
                "JobRole": "Sales Representative", "JobSatisfaction": 1, "MaritalStatus": "Single",
                "MonthlyIncome": 2800, "MonthlyRate": 12000, "NumCompaniesWorked": 5,
                "OverTime": "Yes", "PercentSalaryHike": 12, "PerformanceRating": 3,
                "RelationshipSatisfaction": 2, "StockOptionLevel": 0, "TotalWorkingYears": 4,
                "TrainingTimesLastYear": 2, "WorkLifeBalance": 1, "YearsAtCompany": 2,
                "YearsInCurrentRole": 2, "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 2,
            }
        }
    }


class Prediction(BaseModel):
    employee_name: Optional[str]
    employee_id: Optional[int]
    attrition_probability: float = Field(..., description="Model probability of leaving, 0-1")
    risk_tier: str = Field(..., description="High | Medium | Low")
    will_attrite: bool = Field(..., description="True if probability >= decision threshold")
    threshold: float
    risk_factors: list[str] = Field(..., description="Individual drivers that apply to this employee")
    model_name: str


#  business logic
HIGH_RISK_ROLES = {"Sales Representative", "Laboratory Technician", "Human Resources"}

# Ordered by the model's global driver ranking + EDA direction.
# Each rule: (message, predicate on the employee dict). Transparent, per-employee explanation
# layer that gives the downstream AI agent concrete factors to reason over.
def _risk_factors(e: dict) -> list[str]:
    inc_med = MEDIANS.get("MonthlyIncome", 4900)
    rules = [
        ("Works overtime",                        e["OverTime"] == "Yes"),
        ("Has changed employers often (>=5)",     e["NumCompaniesWorked"] >= 5),
        ("Single",                                e["MaritalStatus"] == "Single"),
        ("Low job satisfaction (<=2)",            e["JobSatisfaction"] <= 2),
        ("Travels frequently for business",       e["BusinessTravel"] == "Travel_Frequently"),
        ("In a high-attrition job role",          e["JobRole"] in HIGH_RISK_ROLES),
        ("Low job involvement (<=2)",             e["JobInvolvement"] <= 2),
        ("No stock options",                      e["StockOptionLevel"] == 0),
        ("Below-median monthly income",           e["MonthlyIncome"] < inc_med),
        ("Short tenure at company (<=2 yrs)",     e["YearsAtCompany"] <= 2),
        ("Poor work-life balance (<=2)",          e["WorkLifeBalance"] <= 2),
        ("Younger employee (<34)",                e["Age"] < 34),
    ]
    fired = [msg for msg, cond in rules if cond]
    return fired[:6] if fired else ["No major individual risk factors identified"]


def _tier(p: float) -> str:
    if p >= TIER_CUTS["High"]:
        return "High"
    if p >= TIER_CUTS["Medium"]:
        return "Medium"
    return "Low"


def _predict_one(emp: Employee) -> Prediction:
    data = emp.model_dump()
    row = pd.DataFrame([{c: data[c] for c in FEATURE_ORDER}])   # exact training column order
    prob = float(PIPELINE.predict_proba(row)[0, 1])
    return Prediction(
        employee_name=emp.employee_name,
        employee_id=emp.employee_id,
        attrition_probability=round(prob, 4),
        risk_tier=_tier(prob),
        will_attrite=prob >= THRESHOLD,
        threshold=THRESHOLD,
        risk_factors=_risk_factors(data),
        model_name=META["model_name"],
    )


# endpoints
@app.get("/", tags=["meta"])
def root():
    """Service metadata — model, threshold, and held-out performance."""
    return {
        "service": "Employee Attrition Prediction API",
        "status": "ok",
        "model": META["model_name"],
        "decision_threshold": THRESHOLD,
        "risk_tiers": TIER_CUTS,
        "test_metrics": META["test_metrics"],
        "n_features": len(FEATURE_ORDER),
        "docs": "/docs",
    }


@app.get("/health", tags=["meta"])
def health():
    """Liveness probe for the cloud platform and the n8n workflow."""
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction, tags=["prediction"])
def predict(employee: Employee):
    """Score a single employee and return probability, risk tier, and applicable risk factors."""
    return _predict_one(employee)


@app.post("/predict/batch", response_model=list[Prediction], tags=["prediction"])
def predict_batch(employees: list[Employee]):
    """Score many employees in one call (convenience for bulk workforce scans)."""
    return [_predict_one(e) for e in employees]
