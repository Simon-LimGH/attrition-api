# Employee Attrition Prediction API

A FastAPI service that exposes the trained attrition model as a REST API so predictions
can be requested programmatically by the n8n agentic workflow.

## What it serves

The deployed artifact `attrition_pipeline.joblib` is a **complete scikit-learn `Pipeline`**
(preprocessing + Logistic Regression). Because preprocessing travels inside the artifact, the API
transforms each request **identically to training** — no training/serving skew. `model_metadata.json`
supplies the decision threshold, risk-tier cut-points, feature order, and per-feature medians.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/` | Service metadata: model, threshold, tiers, held-out test metrics |
| `GET`  | `/health` | Liveness probe (`{"status":"ok"}`) |
| `POST` | `/predict` | Score one employee |
| `POST` | `/predict/batch` | Score a list of employees |
| `GET`  | `/docs` | Interactive Swagger UI (auto-generated) |

### `POST /predict`

**Request** (30 employee attributes; `employee_name` / `employee_id` are optional and echoed back):

```json
{
  "employee_name": "Jordan Lee",
  "employee_id": 1042,
  "Age": 29, "BusinessTravel": "Travel_Frequently", "DailyRate": 800, "Department": "Sales",
  "DistanceFromHome": 15, "Education": 3, "EducationField": "Marketing",
  "EnvironmentSatisfaction": 1, "Gender": "Male", "HourlyRate": 60, "JobInvolvement": 2,
  "JobLevel": 1, "JobRole": "Sales Representative", "JobSatisfaction": 1,
  "MaritalStatus": "Single", "MonthlyIncome": 2800, "MonthlyRate": 12000,
  "NumCompaniesWorked": 5, "OverTime": "Yes", "PercentSalaryHike": 12, "PerformanceRating": 3,
  "RelationshipSatisfaction": 2, "StockOptionLevel": 0, "TotalWorkingYears": 4,
  "TrainingTimesLastYear": 2, "WorkLifeBalance": 1, "YearsAtCompany": 2,
  "YearsInCurrentRole": 2, "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 2
}
```

**Response**:

```json
{
  "employee_name": "Jordan Lee",
  "employee_id": 1042,
  "attrition_probability": 0.9991,
  "risk_tier": "High",
  "will_attrite": true,
  "threshold": 0.417,
  "risk_factors": [
    "Works overtime", "Has changed employers often (>=5)", "Single",
    "Low job satisfaction (<=2)", "Travels frequently for business",
    "In a high-attrition job role"
  ],
  "model_name": "Logistic Regression"
}
```

`risk_factors` is a transparent, per-employee explanation layer (rules ordered by the model's global
driver ranking from Task 2c), giving the downstream AI agent concrete factors to reason over.

**Sample `curl`:**

```bash
curl -X POST https://<your-app>.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d @sample_employee.json
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# open http://127.0.0.1:8000/docs
```

## Deploy on Render (free tier)

1. Create a **GitHub repository whose root is the contents of this `api/` folder** (so `main.py`,
   `requirements.txt`, `render.yaml`, and the two model files sit at the repo root).
2. In Render, **New → Blueprint** and select the repo (uses `render.yaml`), **or** create a
   **Web Service** manually with:
   - **Root Directory:** *(leave blank — repo root)*
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Render builds and returns a public URL, e.g. `https://attrition-prediction-api.onrender.com`.
   The n8n **HTTP Request** node (Task 2e) posts to `<url>/predict`.

> Note: the free tier sleeps after inactivity, so the first request after idle takes ~30–60 s to
> cold-start. The n8n workflow tolerates this with a request timeout / retry.

## Files

| File | Role |
|---|---|
| `main.py` | FastAPI application |
| `attrition_pipeline.joblib` | Trained preprocessing + model pipeline (Task 2c) |
| `model_metadata.json` | Threshold, tier cuts, feature schema, medians, top drivers |
| `requirements.txt` | Pinned dependencies (scikit-learn 1.7.2 matches the training env) |
| `render.yaml` | Render blueprint |
| `sample_employee.json` | Example request body |
