import numpy as np
import pytest

from urlguard.baselines import build_tfidf, train_baselines
from urlguard.features import lex_matrix
from urlguard.lexical_models import train_lexical_models
from urlguard.stacking import fit_stacked, stacked_scores


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    good = [f"shop{i}.com/item/{i % 9}" for i in range(200)]
    bad = [f"a{rng.integers(10**5)}-secure-login.xyz/verify{i}.php" for i in range(200)]
    urls = good + bad
    y = np.array([0] * 200 + [1] * 200)
    groups = np.array([u.split("/")[0] for u in urls])
    tfidf = build_tfidf(5000)
    return urls, y, groups, tfidf.fit_transform(urls), lex_matrix(urls, dtype=np.float32)


def test_baselines_separate_an_easy_problem(toy):
    urls, y, groups, Xt, Xl = toy
    for name, (model, scorer) in train_baselines(Xt, y).items():
        assert (model.predict(Xt) == y).mean() > 0.95, name
        assert scorer(Xt).shape == (len(y),)


def test_lexical_models_include_random_forest(toy):
    urls, y, groups, Xt, Xl = toy
    models = train_lexical_models(Xl, y)
    assert "RandomForest (lexical)" in models
    model, scorer = models["RandomForest (lexical)"]
    assert (model.predict(Xl) == y).mean() > 0.95


def test_stacked_model_scores_are_probabilities(toy):
    urls, y, groups, Xt, Xl = toy
    svm, gbm = fit_stacked(Xt, Xl, y, groups, n_splits=4)
    p = stacked_scores(svm, gbm, Xt, Xl)
    assert p.shape == (len(y),) and ((p >= 0) & (p <= 1)).all()
    assert ((p >= 0.5).astype(int) == y).mean() > 0.95
