"""Character n-gram TF-IDF features and the linear/probabilistic baselines."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def build_tfidf(max_features=150_000, ngram_range=(3, 4)):
    """Character n-gram vectorizer. float32 keeps memory manageable on ~650k URLs."""
    return TfidfVectorizer(analyzer="char", ngram_range=ngram_range,
                           max_features=max_features, sublinear_tf=True,
                           dtype=np.float32)


def train_baselines(X, y, svm_c=1.0):
    """Fit logistic regression, linear SVM and naive Bayes on TF-IDF features.

    Returns ``{name: (model, scorer)}`` where ``scorer(X)`` gives a continuous
    malicious score suitable for ROC-AUC.
    """
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
    svm = LinearSVC(C=svm_c, class_weight="balanced", max_iter=5000).fit(X, y)
    nb = MultinomialNB(alpha=0.1).fit(X, y)
    return {
        "LogReg (TF-IDF)": (lr, lambda A: lr.predict_proba(A)[:, 1]),
        "LinearSVM (TF-IDF)": (svm, svm.decision_function),
        "NaiveBayes (TF-IDF)": (nb, lambda A: nb.predict_proba(A)[:, 1]),
    }
