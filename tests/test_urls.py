from urlguard.urls import clean_url, split_url


def test_clean_url_strips_scheme_and_www():
    assert clean_url("HTTPS://www.Example.com/a") == "Example.com/a"
    assert clean_url("  http://a.com  ") == "a.com"


def test_clean_url_leaves_other_schemes_alone():
    assert clean_url("ftp://a.com/x") == "ftp://a.com/x"


def test_split_url_full():
    assert split_url("https://a.b.com:8080/p/q?x=1#frag") == (
        "https", "a.b.com:8080", "/p/q", "x=1")


def test_split_url_without_scheme_defaults_to_http():
    assert split_url("a.com/x") == ("http", "a.com", "/x", "")


def test_split_url_host_only():
    assert split_url("a.com") == ("http", "a.com", "", "")


def test_split_url_never_raises_on_malformed_input():
    for u in ["http://[bad/path", "a]b.com/x", "http://[::1", "", "//", "http://"]:
        scheme, host, path, query = split_url(u)
        assert isinstance(host, str)
