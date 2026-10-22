"""Endpoint tests against the committed model.

The probabilities pinned in ``test_features_reproduce_the_exported_model`` are
what the model gave when it was first exported, so they fail if the feature code
in ``urlguard`` ever drifts from the code the model was trained with.
"""
import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def predict(url):
    r = client.post("/predict", json={"url": url})
    assert r.status_code == 200, r.text
    return r.json()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_obvious_phishing_is_malicious():
    d = predict("paypa1-login.xyz/verify")
    assert d["verdict"] == "malicious" and d["probability"] > 0.9
    assert {"login", "verify"} <= set(d["signals"]["suspicious_words"])


def test_popular_sites_are_benign():
    assert predict("en.wikipedia.org")["verdict"] == "benign"
    d = predict("github.com/c-tasnia")
    assert d["verdict"] == "benign" and d["signals"]["in_popular_list"] is True


def test_lookalike_domain_is_flagged():
    assert predict("gooogle.com/accounts/login")["verdict"] == "malicious"


def test_ip_address_host_is_reported():
    d = predict("http://192.168.4.5/update.php")
    assert d["signals"]["ip_address_host"] is True and d["verdict"] == "malicious"


def test_scheme_and_www_do_not_change_the_score():
    a = predict("https://www.example-shop.com/a")
    b = predict("example-shop.com/a")
    assert a["probability"] == b["probability"]
    assert a["cleaned"] == "example-shop.com/a"


@pytest.mark.parametrize("url,expected", [
    ("paypa1-login.xyz/verify", 0.9849),
    ("bkash-secure-login.xyz/verify", 0.9883),
    ("github.com/c-tasnia", 0.0859),
    ("en.wikipedia.org", 0.0091),
])
def test_features_reproduce_the_exported_model(url, expected):
    assert predict(url)["probability"] == pytest.approx(expected, abs=2e-3)


@pytest.mark.parametrize("bad", ["", "a b", "two words.com/x", "   "])
def test_invalid_input_is_rejected(bad):
    assert client.post("/predict", json={"url": bad}).status_code == 422


def test_overlong_input_is_rejected():
    assert client.post("/predict", json={"url": "a" * 2049}).status_code == 422


def test_malformed_url_does_not_crash():
    assert predict("http://[broken/url")["verdict"] in {"malicious", "benign"}


def test_home_page_is_served():
    r = client.get("/")
    assert r.status_code == 200 and "URL Guard" in r.text


def test_favicon_does_not_404():
    assert client.get("/favicon.ico").status_code == 204
