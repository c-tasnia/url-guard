# URL Guard

Detect malicious URLs (phishing, malware hosting, defacement) from the URL text alone, without opening the link. A trained model, the training pipeline that produces it, and a small web app to try it.

Only the string is used: no page content, WHOIS, DNS or reputation lookups. That makes it fast and safe to run on any link, and it also limits how accurate it can be (see [docs/evaluation.md](docs/evaluation.md)).

## How it works

```
URL -> clean (drop scheme and www) -+-> char n-gram TF-IDF -> LinearSVC score --+
                                    |                                          +-> LightGBM -> P(malicious)
                                    +-> 27 lexical + popularity features ------+
```

- **Char n-gram SVM:** reads the URL as text and gives one score.
- **Lexical features:** lengths, character counts, entropy, IP host, suspicious words, risky file extension, and whether the domain is in the Tranco top 100k.
- **LightGBM:** combines the SVM score (out-of-fold at training time) with the lexical features.

Training adds popular and small-site domains from the Tranco list as extra benign rows, and synthetic typosquats (`paypa1.com`, `gooogle.com`) as extra malicious rows.

## Run the website

Requires Python 3.12 or 3.13 (the pinned scikit-learn and LightGBM versions need a Python they ship builds for).

```
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --port 8000
```

Open http://localhost:8000.

### API

```
POST /predict   {"url": "paypa1-login.xyz/verify"}
-> {"verdict": "malicious", "probability": 0.98, "threshold": 0.5,
    "cleaned": "paypa1-login.xyz/verify", "signals": {...}}

GET  /health
```

The decision threshold is stored in the model file; override it without retraining with `THRESH=0.6 uvicorn app:app`. Point at another model with `MODEL_PATH=...`.

## Train

The dataset is not in this repository. It needs a URL column and a label column; in the dataset this project used, `0` means malicious.

```
pip install -r requirements-train.txt
python train.py --data testingurldataset.csv --malicious-label 0
```

This downloads the Tranco list (or use `--tranco-file`), trains the baselines and the stacked model on a domain-grouped split, prints metrics on all test rows and on the original rows only, and writes `model/url_model.joblib` and `model/metrics.json`. `--quick` trains only the stacked model.

The committed `model/url_model.joblib` was exported from the original Colab notebook run. `train.py` is the reproducible script version of that pipeline; it samples Tranco domains randomly, so a fresh run will not reproduce the committed model exactly.

Optional: `pip install xgboost` adds an XGBoost model to the comparison.

## Tests

```
pip install -r requirements-dev.txt
python -m pytest
```

`tests/test_api.py` pins the probabilities the committed model gave for four URLs when it was exported, so it fails if the feature code in `urlguard/` drifts from what the model was trained with.

## Layout

```
app.py              FastAPI app and /predict endpoint
static/index.html   single-page frontend
train.py            training CLI
urlguard/
  urls.py           URL cleaning and splitting (no urlparse)
  popularity.py     Tranco rank features
  features.py       lexical features, shared by training and serving
  data.py           dataset loading and label normalisation
  tranco.py         extra benign rows from popular domains
  typosquat.py      synthetic lookalike malicious rows
  dataset.py        assembles the training table
  split.py          domain-grouped train/test split
  baselines.py      TF-IDF + LogReg / SVM / Naive Bayes
  lexical_models.py random forest / XGBoost on lexical features
  stacking.py       out-of-fold stacked SVM + LightGBM
  evaluate.py       metrics and threshold selection
model/              trained model
docs/evaluation.md  how to evaluate, and what went wrong before
```

## Security note

`model/url_model.joblib` is a pickle. Loading a pickle can run arbitrary code, so only load model files you exported yourself or trust.

## Deploy

Any host that runs a Python web process works. Build with `pip install -r requirements.txt` and start with `uvicorn app:app --host 0.0.0.0 --port $PORT` (Render, Railway) or port 7860 on Hugging Face Spaces (Docker).

## Limitations

This is a statistical estimate from the URL text alone, not a security guarantee. Do not use it as the only check. See [docs/evaluation.md](docs/evaluation.md) for known failure modes, including false positives on personal sites hosted on shared platforms.
