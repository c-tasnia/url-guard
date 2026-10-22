import numpy as np

from urlguard.features import FEATURE_NAMES, lex_matrix, lexical
from urlguard.popularity import build_rank

IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def test_vector_length_matches_names():
    assert len(lexical("a.com/x")) == len(FEATURE_NAMES) == 27


def test_matrix_shape_and_dtype():
    m = lex_matrix(["a.com", "b.org/x?y=1"], dtype=np.float32)
    assert m.shape == (2, len(FEATURE_NAMES))
    assert m.dtype == np.float32


def test_suspicious_words_counted():
    v = lexical("paypa1-login.xyz/verify")
    assert v[IDX["has_suspicious_word"]] == 1
    assert v[IDX["suspicious_word_count"]] == 2
    assert v[IDX["hyphens"]] == 1


def test_ip_host_flag():
    assert lexical("192.168.4.5/update.php")[IDX["host_is_ip"]] == 1
    assert lexical("example.com")[IDX["host_is_ip"]] == 0


def test_risky_extension_flag():
    assert lexical("a.com/setup.exe")[IDX["risky_extension"]] == 1
    assert lexical("a.com/index.html")[IDX["risky_extension"]] == 0


def test_malformed_url_does_not_raise():
    assert len(lexical("http://[bad/path")) == len(FEATURE_NAMES)


def test_popularity_features_use_rank():
    rank = build_rank(["github.com"], [100])
    assert lexical("github.com/x", rank)[IDX["in_popular_list"]] == 1
    assert lexical("github.com/x")[IDX["in_popular_list"]] == 0
    assert lexical("githuub.com/x", rank)[IDX["in_popular_list"]] == 0
