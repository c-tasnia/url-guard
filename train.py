"""Train the stacked URL detector and export it for the web app.

Example:
    python train.py --data testingurldataset.csv --malicious-label 0

The Tranco list is downloaded unless --tranco-file is given (the zip from
tranco-list.eu or a plain rank,domain CSV). Use --no-tranco to train on the
source CSV alone.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

from urlguard.baselines import build_tfidf, train_baselines
from urlguard.dataset import build_dataset
from urlguard.evaluate import report, to_markdown
from urlguard.features import lex_matrix
from urlguard.lexical_models import train_lexical_models
from urlguard.split import grouped_split
from urlguard.stacking import fit_stacked, stacked_scores


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV with a URL column and a label column")
    ap.add_argument("--url-col", default="url")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--malicious-label", default="0",
                    help="label value meaning malicious (0 in the original dataset)")
    ap.add_argument("--tranco-file", default=None)
    ap.add_argument("--no-tranco", action="store_true")
    ap.add_argument("--n-popular", type=int, default=100_000)
    ap.add_argument("--n-tail", type=int, default=100_000)
    ap.add_argument("--n-typo", type=int, default=20_000)
    ap.add_argument("--tfidf-features", type=int, default=150_000)
    ap.add_argument("--svm-c", type=float, default=0.3)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="decision threshold stored with the model")
    ap.add_argument("--per-host-cap", type=int, default=10)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--quick", action="store_true",
                    help="skip the baseline and lexical models, train only the stacked model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="model")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()

    data, rank = build_dataset(
        args.data, args.url_col, args.label_col, args.malicious_label,
        tranco_path=args.tranco_file, use_tranco=not args.no_tranco,
        n_popular=args.n_popular, n_tail=args.n_tail, n_typo=args.n_typo,
        seed=args.seed)
    print(data.groupby(["src", "label"]).size())

    train_df, test_df = grouped_split(data, args.test_size, args.per_host_cap, args.seed)
    ytr, yte = train_df["label"].to_numpy(), test_df["label"].to_numpy()
    print(f"train {len(train_df)} rows, test {len(test_df)} rows (no host in both)")

    tfidf = build_tfidf(args.tfidf_features)
    Xtr_t = tfidf.fit_transform(train_df["url"])
    Xte_t = tfidf.transform(test_df["url"])
    Xtr_l = lex_matrix(train_df["url"], rank, np.float32)
    Xte_l = lex_matrix(test_df["url"], rank, np.float32)

    scored = {}   # model name -> (predictions, scores) on the test set
    if not args.quick:
        for name, (model, scorer) in train_baselines(Xtr_t, ytr).items():
            scored[name] = (model.predict(Xte_t), scorer(Xte_t))
        for name, (model, scorer) in train_lexical_models(Xtr_l, ytr, args.seed).items():
            scored[name] = (model.predict(Xte_l), scorer(Xte_l))

    svm, gbm = fit_stacked(Xtr_t, Xtr_l, ytr, train_df["host"].to_numpy(),
                           svm_c=args.svm_c, seed=args.seed)
    p = stacked_scores(svm, gbm, Xte_t, Xte_l)
    scored["Stacked SVM + LightGBM"] = ((p >= args.threshold).astype(int), p)

    orig = (test_df["src"] == "orig").to_numpy()
    rows_all = [report(n, yte, pr, sc) for n, (pr, sc) in scored.items()]
    rows_orig = []
    if orig.any() and len(set(yte[orig])) == 2:
        rows_orig = [report(n, yte, pr, sc, mask=orig, verbose=False)
                     for n, (pr, sc) in scored.items()]

    print("\nAll test rows (includes easy added popular-domain and typosquat rows):")
    print(to_markdown(rows_all))
    if rows_orig:
        print("\nOriginal dataset rows only (the headline number):")
        print(to_markdown(rows_orig))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"tfidf": tfidf, "svm": svm, "gbm": gbm, "rank": rank,
                 "thresh": args.threshold}, out / "url_model.joblib", compress=3)
    (out / "metrics.json").write_text(json.dumps(
        {"all_test_rows": rows_all, "original_rows_only": rows_orig,
         "config": vars(args)}, indent=2))
    print(f"\nsaved {out / 'url_model.joblib'} and metrics.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
