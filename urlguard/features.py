"""Hand-built lexical features for a URL.

The same code is used by the training pipeline and by the web app, so a model
trained here sees exactly the features it was trained on when serving.
"""
import math
import re
from collections import Counter

import numpy as np

from .popularity import pop_feat
from .urls import split_url

SUSPICIOUS = ["login", "verify", "update", "secure", "account", "bank", "free",
              "bonus", "confirm", "signin", "paypal", "password"]

FEATURE_NAMES = [
    "url_len", "host_len", "path_len", "query_len",
    "dots", "hyphens", "underscores", "slashes", "question_marks", "equals",
    "ampersands", "ats", "percents", "double_slashes",
    "digit_ratio", "alpha_ratio",
    "host_dots", "host_is_ip", "is_https",
    "has_suspicious_word", "suspicious_word_count",
    "host_entropy", "path_entropy",
    "long_tld", "risky_extension",
    "in_popular_list", "popularity_score",
]


def entropy(s):
    """Shannon entropy of a string in bits per character."""
    if not s:
        return 0.0
    c = Counter(s)
    return -sum(n / len(s) * math.log2(n / len(s)) for n in c.values())


def lexical(url, rank=None):
    """Feature vector for one URL, in ``FEATURE_NAMES`` order."""
    rank = rank or {}
    url = str(url).strip()
    scheme, host, path, query = split_url(url)
    low = url.lower()
    return [
        len(url), len(host), len(path), len(query),
        url.count("."), url.count("-"), url.count("_"), url.count("/"),
        url.count("?"), url.count("="), url.count("&"), url.count("@"),
        url.count("%"), url.count("//"),
        sum(c.isdigit() for c in url) / max(len(url), 1),
        sum(c.isalpha() for c in url) / max(len(url), 1),
        host.count("."),
        int(bool(re.fullmatch(r"[\d.:]+", host))),
        int(scheme == "https"),
        int(any(w in low for w in SUSPICIOUS)),
        sum(w in low for w in SUSPICIOUS),
        entropy(host), entropy(path),
        int(len(host.split(".")[-1]) > 4) if host else 0,
        int(bool(re.search(r"\.(exe|zip|php|js|apk|scr)$", path.lower()))),
        *pop_feat(host, rank),
    ]


def lex_matrix(urls, rank=None, dtype=np.float64):
    """Feature matrix with one row per URL."""
    return np.array([lexical(u, rank) for u in urls], dtype=dtype)
