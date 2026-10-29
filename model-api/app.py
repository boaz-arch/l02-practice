import os
import random

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

APP_VERSION = os.getenv("APP_VERSION", "dev")
ERROR_RATE = float(os.getenv("ERROR_RATE", "0"))  # used later for chaos drills
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

app = FastAPI(title="model-api")
model = joblib.load(MODEL_PATH)

class Features(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}

@app.post("/predict")
def predict(req: Features):
    if random.random() < ERROR_RATE:
        raise HTTPException(status_code=500, detail="injected failure")
    if len(req.features) != 4:
        raise HTTPException(status_code=422, detail="expected 4 features")
    prediction = int(model.predict([req.features])[0])
    return {"prediction": prediction, "version": APP_VERSION}
