# URL Guard

Detect malicious URLs (phishing, malware hosting, defacement) from the URL text alone, without opening the link.

## Goal

Given a URL, estimate the probability that it is malicious. Only the string is used: no page content, WHOIS, DNS or reputation lookups. That makes it fast and safe to run on any link, and also limits how accurate it can be.

## Dataset conventions

The training CSV has a `url` column and a `label` column. In this dataset **`0` means malicious and `1` means benign**, the opposite of what most code assumes. The code takes the malicious label as an explicit setting instead of guessing.

The CSV itself is not stored in this repository.

## Status

Work in progress: feature extraction, training pipeline and a small web app are being added.
