from urlguard.popularity import build_rank, pop_feat, popular_parent

RANK = build_rank(["Wikipedia.org", "github.com"], [10, 100])


def test_subdomain_matches_parent():
    assert popular_parent("en.wikipedia.org", RANK) == "wikipedia.org"


def test_exact_match_and_score():
    assert pop_feat("github.com", RANK) == [1, 4.0]


def test_unknown_domain_scores_zero():
    assert pop_feat("evil.com", RANK) == [0, 0]


def test_lookalike_is_not_matched():
    assert popular_parent("githuub.com", RANK) is None


def test_popular_name_inside_attacker_domain_is_not_matched():
    assert popular_parent("wikipedia.org.evil.com", RANK) is None


def test_port_is_ignored():
    assert popular_parent("github.com:443", RANK) == "github.com"
