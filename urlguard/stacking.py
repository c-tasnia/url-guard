"""Stacked model: a char n-gram SVM score feeds a LightGBM model with lexical features.

The SVM score given to LightGBM at training time must be out-of-fold. If it is
computed on data the SVM was fitted on, it looks far more reliable than it
will on new sites and LightGBM over-trusts it. The folds are grouped by host
for the same reason as the train/test split: random folds leak domains.
"""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.svm import LinearSVC


def fit_stacked(X_tfidf, X_lex, y, groups, svm_c=0.3, n_splits=4, seed=42):
    """Return ``(svm, gbm)``. ``groups`` is the host of each training row."""
    svm = LinearSVC(C=svm_c, class_weight="balanced", max_iter=5000)
    oof = cross_val_predict(svm, X_tfidf, y, cv=GroupKFold(n_splits=n_splits),
                            groups=groups, method="decision_function")
    svm.fit(X_tfidf, y)
    gbm = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=seed, verbose=-1)
    gbm.fit(np.column_stack([X_lex, oof]), y)
    return svm, gbm


def stacked_scores(svm, gbm, X_tfidf, X_lex):
    """Probability of malicious for each row."""
    stacked = np.column_stack([X_lex, svm.decision_function(X_tfidf)])
    return gbm.predict_proba(stacked)[:, 1]
