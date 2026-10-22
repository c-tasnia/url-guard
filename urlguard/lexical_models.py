"""Tree models on the hand-built lexical features."""
from sklearn.ensemble import RandomForestClassifier


def train_lexical_models(X, y, seed=42):
    """Fit a random forest, and XGBoost if it is installed.

    Returns ``{name: (model, scorer)}`` like ``baselines.train_baselines``.
    """
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                n_jobs=-1, random_state=seed).fit(X, y)
    models = {"RandomForest (lexical)": (rf, lambda A: rf.predict_proba(A)[:, 1])}
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return models
    xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", random_state=seed).fit(X, y)
    models["XGBoost (lexical)"] = (xgb, lambda A: xgb.predict_proba(A)[:, 1])
    return models
