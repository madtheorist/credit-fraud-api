import os
import joblib
import numpy as np
import pandas as pd
from typing import List
import sklearn.pipeline
from sklearn.pipeline import Pipeline
from pipeline import Preprocessor
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


model_path = os.path.join('models', 'rf_model_final.pkl')
preprocessor_path = os.path.join('models', 'preprocessor.pkl')
model: RandomForestClassifier = joblib.load(model_path)
preprocessor: Preprocessor = joblib.load(preprocessor_path)
PROBABILITY_THRESHOLD = 0.875

app = FastAPI()

class InputData(BaseModel):
    amt: float
    hour: int
    time_since_last_minutes: float
    category: str # e.g. "shopping_net"

@app.post("/predict")
def predict(data: List[InputData]):
    try:
        X = pd.DataFrame([item.model_dump() for item in data])
        X_processed = preprocessor.transform(X) # apply transformations
        predictions = model.predict_proba(X_processed)[:, 1] # get probability for class 1

        # if probability of fraud is greater than threshold, return 1; otherwise return 0
        y_pred = (predictions > PROBABILITY_THRESHOLD).astype(int)
        return {"predictions": y_pred.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))