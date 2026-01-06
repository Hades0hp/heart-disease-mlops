from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
import time
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

request_count = defaultdict(int)

app = FastAPI(title="Heart Disease Prediction API")

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    request_count[request.url.path] += 1

    logger.info(
        f"Method={request.method} "
        f"Path={request.url.path} "
        f"Status={response.status_code} "
        f"Latency={process_time:.4f}s"
    )

    return response

# Load model from file (inside container)
model = joblib.load("model.pkl")

class PatientInput(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(input: PatientInput):
    df = pd.DataFrame([input.dict()])
    proba = model.predict_proba(df)[0][1]
    prediction = int(proba >= 0.5)
    return {
        "prediction": prediction,
        "confidence": round(float(proba), 4)
    }

@app.get("/metrics")
def metrics():
    return {
        "request_count": dict(request_count)
    }