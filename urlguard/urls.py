"""URL cleaning and splitting.

``urllib.parse.urlparse`` raises ``ValueError`` on malformed URLs (for example a
stray ``[`` in a phishing link), so URLs are split with plain string operations
instead. Nothing in this module ever raises.
"""
import re

_SCHEME = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*)://(.*)$")


def clean_url(url):
    """Drop the scheme and a leading ``www.``.

    In the source dataset 69% of malicious URLs started with ``http`` against
    8% of benign ones, so a model that sees the scheme simply learns that
    shortcut instead of anything about the URL itself.
    """
    u = str(url).strip()
    u = re.sub(r"^https?://", "", u, flags=re.I)
    return re.sub(r"^www\.", "", u, flags=re.I)


def split_url(url):
    """Return ``(scheme, host, path, query)``; the scheme defaults to ``http``."""
    url = str(url).strip()
    m = _SCHEME.match(url)
    scheme, rest = (m.group(1).lower(), m.group(2)) if m else ("http", url)
    host_end = len(rest)
    for ch in "/?#":
        i = rest.find(ch)
        if i != -1:
            host_end = min(host_end, i)
    host = rest[:host_end]
    tail = rest[host_end:].split("#", 1)[0]
    path, _, query = tail.partition("?")
    return scheme, host, path, query
