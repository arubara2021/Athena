from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import parse_qsl, urljoin, urlencode, urlsplit, urlunsplit

DEFAULT_SCHEME = "https"
ALLOWED_SCHEMES = {"http", "https"}

TRACKING_PARAMETERS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "igshid",
    "ref",
    "ref_src",
}


def normalize_url(
    url: str,
    remove_tracking: bool = True,
    remove_fragment: bool = False,
    lowercase_host: bool = True,
    default_scheme: str = DEFAULT_SCHEME,
) -> str:
    if not isinstance(url, str):
        raise ValueError("url must be a string")

    candidate = url.strip()

    if not candidate:
        raise ValueError("url cannot be empty")

    if candidate.startswith("//"):
        candidate = f"{default_scheme}:{candidate}"
    elif "://" not in candidate:
        candidate = f"{default_scheme}://{candidate}"

    parts = urlsplit(candidate)
    scheme = parts.scheme.lower()
    netloc = parts.netloc.strip()

    if lowercase_host:
        netloc = netloc.lower()

    if ":" in netloc:
        host, _, port = netloc.rpartition(":")

        if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
            netloc = host

    path = parts.path or "/"

    query_pairs = parse_qsl(parts.query, keep_blank_values=True)

    if remove_tracking:
        query_pairs = [
            (key, value)
            for key, value in query_pairs
            if key.lower() not in TRACKING_PARAMETERS
        ]

    query = urlencode(query_pairs, doseq=True)
    fragment = "" if remove_fragment else parts.fragment

    return urlunsplit((scheme, netloc, path, query, fragment))


def is_valid_url(url: str, schemes: Iterable[str] = ALLOWED_SCHEMES) -> bool:
    try:
        normalized = normalize_url(url)
        parts = urlsplit(normalized)
        return parts.scheme in set(schemes) and bool(parts.netloc)
    except Exception:
        return False


def get_domain(url: str) -> str:
    normalized = normalize_url(url)
    return urlsplit(normalized).netloc


def get_base_url(url: str) -> str:
    parts = urlsplit(normalize_url(url))
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def ensure_absolute_url(url: str, base_url: str) -> str:
    if not isinstance(url, str):
        raise ValueError("url must be a string")

    if not isinstance(base_url, str):
        raise ValueError("base_url must be a string")

    candidate = url.strip()

    if not candidate:
        raise ValueError("url cannot be empty")

    if is_valid_url(candidate):
        return normalize_url(candidate)

    joined = urljoin(base_url.strip(), candidate)
    return normalize_url(joined)


def get_url_fingerprint(url: str) -> str:
    normalized = normalize_url(
        url,
        remove_tracking=True,
        remove_fragment=True,
    )
    return normalized.rstrip("/").lower()