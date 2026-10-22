import numpy as np

from urlguard.dataset import build_dataset
from urlguard.tranco import (SUBDOMAINS, benign_paths_from, load_tranco,
                             make_benign, split_popular_tail)
from urlguard.typosquat import make_typosquats, typo
from urlguard.data import load_dataset


def test_load_tranco_from_plain_csv(tiny_tranco):
    t = load_tranco(tiny_tranco)
    assert list(t.columns) == ["rank", "domain"]
    assert len(t) == 500


def test_split_popular_tail_is_disjoint(tiny_tranco):
    popular, tail = split_popular_tail(load_tranco(tiny_tranco), 100, 50, seed=1)
    assert len(popular) == 100 and len(tail) == 50
    assert not set(popular["domain"]) & set(tail["domain"])


def test_make_benign_labels_and_variants(tiny_tranco):
    rng = np.random.default_rng(1)
    domains = [f"popular{i}.com" for i in range(200)]
    df = make_benign(domains, ["/about", "/docs/a"], rng, "tranco", subdomain_share=0.5)
    assert (df["label"] == 0).all() and (df["src"] == "tranco").all()
    assert df["url"].str.startswith(tuple(SUBDOMAINS)).any()
    assert df["url"].str.contains("/about|/docs/a").any()


def test_benign_paths_come_from_benign_rows(tiny_csv):
    paths = benign_paths_from(load_dataset(tiny_csv))
    assert paths and all(p.startswith("/") for p in paths)
    assert not any("verify.php" in p for p in paths)


def test_typo_changes_the_name_and_keeps_the_tld():
    rng = np.random.default_rng(3)
    for _ in range(50):
        t = typo("paypal.com", rng)
        assert t is None or (t != "paypal.com" and t.endswith(".com"))
    assert typo("ab.com", rng) is None


def test_typosquats_never_reuse_a_real_popular_domain():
    rng = np.random.default_rng(4)
    domains = [f"popular{i}.com" for i in range(300)]
    known = set(domains)
    df = make_typosquats(domains, known, rng, n=300)
    assert (df["label"] == 1).all() and len(df) > 100
    assert not set(u.split("/")[0] for u in df["url"]) & known


def test_build_dataset_combines_sources(tiny_csv, tiny_tranco):
    data, rank = build_dataset(tiny_csv, tranco_path=tiny_tranco,
                               n_popular=100, n_tail=100, n_typo=100)
    assert set(data["src"]) == {"orig", "tranco", "tranco_small", "typosquat"}
    assert data["url"].is_unique
    assert len(rank) == 100 and rank["popular1.com"] == 1
    assert (data.loc[data["src"] == "tranco", "label"] == 0).all()
    assert (data.loc[data["src"] == "typosquat", "label"] == 1).all()
    assert (data["src"] == "orig").sum() == 601


def test_build_dataset_without_tranco_is_just_the_source(tiny_csv):
    data, rank = build_dataset(tiny_csv, use_tranco=False)
    assert set(data["src"]) == {"orig"} and rank == {}
