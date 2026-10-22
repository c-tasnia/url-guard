from urlguard.data import load_dataset, normalise_labels


def test_zero_is_malicious_by_default():
    assert list(normalise_labels(["0", "1", "0.0", 0, 1])) == [1, 0, 1, 1, 0]


def test_other_label_conventions():
    assert list(normalise_labels(["phishing", "benign", "Phishing "],
                                 malicious_label="phishing")) == [1, 0, 1]
    assert list(normalise_labels([1, 0, 1], malicious_label=1)) == [1, 0, 1]


def test_load_dataset_cleans_urls_and_drops_missing(tiny_csv):
    df = load_dataset(tiny_csv)
    assert list(df.columns) == ["url", "label", "src"]
    assert (df["src"] == "orig").all()
    assert len(df) == 601                       # the row with a missing URL is dropped
    assert not df["url"].str.startswith(("http://", "https://", "www.")).any()


def test_load_dataset_flips_source_labels_to_one_is_malicious(tiny_csv):
    df = load_dataset(tiny_csv)
    assert df.loc[df["url"].str.contains("login.xyz"), "label"].eq(1).all()
    assert df.loc[df["url"].str.contains("site"), "label"].eq(0).all()
