"""URL Guard API.

Run:  uvicorn app:app --port 8000
"""
import os
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from urlguard.features import SUSPICIOUS, lexical
from urlguard.popularity import popular_parent
from urlguard.urls import clean_url, split_url

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE / "model" / "url_model.joblib"))

_m = joblib.load(MODEL_PATH)
tfidf, svm, gbm, rank = _m["tfidf"], _m["svm"], _m["gbm"], _m["rank"]
THRESH = float(os.environ.get("THRESH", _m.get("thresh", 0.5)))

app = FastAPI(title="URL Guard")


class Req(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


def score_url(raw_url):
    """Stacked model: char n-gram SVM score + lexical features -> LightGBM.

    Returns the cleaned URL, the probability of malicious, and a few readable
    signals about the URL (these explain the page, they are not the model's
    reasoning).
    """
    u = clean_url(raw_url)
    svm_score = svm.decision_function(tfidf.transform([u]))[0]
    x = np.array([lexical(u, rank) + [svm_score]], dtype=float)
    p = float(gbm.predict_proba(x)[0, 1])
    host = split_url(u)[1]
    signals = {
        "host": host,
        "in_popular_list": popular_parent(host, rank) is not None,
        "suspicious_words": [w for w in SUSPICIOUS if w in u.lower()],
        "ip_address_host": bool(re.fullmatch(r"[\d.:]+", host)),
        "hyphens_in_host": host.count("-"),
        "length": len(u),
    }
    return u, p, signals


@app.post("/predict")
def predict(req: Req):
    url = req.url.strip()
    if not url or any(c.isspace() for c in url):
        return JSONResponse({"error": "Enter a single URL with no spaces."}, status_code=422)
    cleaned, p, signals = score_url(url)
    return {
        "input": url,
        "cleaned": cleaned,
        "probability": round(p, 4),
        "threshold": THRESH,
        "verdict": "malicious" if p >= THRESH else "benign",
        "signals": signals,
    }


@app.get("/health")
def health():
    return {"ok": True, "threshold": THRESH}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")
