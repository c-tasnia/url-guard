# Evaluation notes

How the model is evaluated and the mistakes that made earlier scores misleading. Read this before comparing numbers.

## 1. Label convention

The source dataset uses `0` for malicious and `1` for benign. Treating `1` as malicious trains a model that is exactly backwards: a phishing-looking URL scored 0.000 and so did github.com. Always pass `--malicious-label` explicitly and sanity-check a few rows from each class.

## 2. The `http://` shortcut

69% of malicious URLs in the source data start with `http`, against 8% of benign URLs. A model that sees the scheme can score well by checking for it. URLs are therefore cleaned (`urlguard.urls.clean_url`) by dropping the scheme and a leading `www.` before any feature is computed, in training and when serving.

## 3. Random split versus domain-grouped split

A random split puts URLs from the same site in both train and test, so a model can score well by memorising domains. `urlguard.split.grouped_split` keeps every host on one side only, so the test set measures performance on sites the model has never seen. It also caps URLs per host (default 10) so a handful of large sites cannot dominate training.

Expect lower scores than a random split gives. They are the ones that mean something.

## 4. Report on the original rows

Training adds popular domains, small-site domains and synthetic typosquats. Those rows are easy, so including them in the test set inflates the score. `train.py` prints two tables:

- **All test rows**, which includes the added rows.
- **Original dataset rows only**, which is the number to report.

## 5. Stacking needs out-of-fold scores

The stacked model feeds a linear SVM score into LightGBM. If that score is computed on the same rows the SVM was trained on, it looks near-perfect and LightGBM learns to over-trust it. `urlguard.stacking.fit_stacked` computes it out-of-fold with `GroupKFold` on the host. With random folds the stacked model scored *below* the SVM it was built on; with grouped folds it scored above it.

## 6. Precomputed feature columns

The source CSV also contains about 33 precomputed numeric columns (`url_length`, `number_of_slash_in_url`, ...). They were computed on the raw URLs, scheme included, so they carry the same shortcut as section 2 and cannot be cleaned afterwards. A model trained on them scored much higher (F1 about 0.95) than any model trained on cleaned URLs, which is a warning sign, not a result. They are not used.

## 7. Choosing the decision threshold

The default threshold is 0.5. Choose a different one on a validation split carved out of the training hosts, then report the test score at that fixed value. Picking the threshold on the test set inflates the score slightly. `urlguard.evaluate.best_threshold` finds the F1-maximising value.

## Known limitations

- **Shared hosting platforms.** Phishing is often hosted on free platforms (vercel.app, blogspot.com, ...). A legitimate personal site on one of them can score as malicious because the training data associates those hosts with abuse. More benign examples from those platforms would help.
- **Synthetic typosquats.** The lookalike domains are generated, not collected. They teach the lookalike pattern but do not cover every real-world trick.
- **URL text only.** Without page content or network signals, a freshly registered but clean-looking malicious domain can look benign, and an unusual legitimate URL can look malicious.
- **Popularity features.** Being in the top 100k domains is a soft signal. It lowers the score for known sites but is not an allowlist, so a compromised popular site can still be flagged.
