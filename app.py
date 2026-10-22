"""URL Guard API.

Run:  uvicorn app:app --port 8000
"""
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from urlguard.features import lexical
from urlguard.urls import clean_url

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE / "model" / "url_model.joblib"))

_m = joblib.load(MODEL_PATH)
tfidf, svm, gbm, rank = _m["tfidf"], _m["svm"], _m["gbm"], _m["rank"]
THRESH = float(os.environ.get("THRESH", _m.get("thresh", 0.5)))

app = FastAPI(title="URL Guard")


class Req(BaseModel):
    url: str


def score_url(raw_url):
    """Stacked model: char n-gram SVM score + lexical features -> LightGBM."""
    u = clean_url(raw_url)
    svm_score = svm.decision_function(tfidf.transform([u]))[0]
    x = np.array([lexical(u, rank) + [svm_score]], dtype=float)
    return u, float(gbm.predict_proba(x)[0, 1])


@app.post("/predict")
def predict(req: Req):
    cleaned, p = score_url(req.url)
    return {"input": req.url, "cleaned": cleaned, "probability": round(p, 4),
            "threshold": THRESH, "verdict": "malicious" if p >= THRESH else "benign"}
