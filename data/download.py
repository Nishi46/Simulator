"""Safe, verified file download.

`fetch` only talks HTTPS to allowlisted hosts, caps size and time, verifies a
pinned sha256, and writes atomically, so `dest` is either absent, the previous
good file, or the new verified file, never a partial download.
"""
from __future__ import annotations

import hashlib
import http.client
import logging
import os
import re
import ssl
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from data.paths import DATA_DIR, safe_join

log = logging.getLogger(__name__)

CHUNK = 64 * 1024
MAX_URL_LEN = 2048
MAX_REDIRECTS = 5
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class DownloadError(ValueError):
    """Raised for any refused or failed download. Messages never contain credentials or query strings."""


@dataclass(frozen=True)
class Policy:
    # Hosts must be confirmed against the official source before being relied on.
    allowed_hosts: tuple[str, ...] = ("gss.norc.org", "www2.census.gov")
    allowed_schemes: tuple[str, ...] = ("https",)
    allowed_ports: tuple[int, ...] = (443,)


DEFAULT_POLICY = Policy()


def _redact(url: str) -> str:
    """scheme://host/path only: drops userinfo, query and fragment."""
    try:
        p = urlsplit(url)
        return f"{p.scheme}://{p.hostname or '?'}{p.path}"
    except ValueError:
        return "<unparseable url>"


def validate_url(url: object, policy: Policy = DEFAULT_POLICY) -> str:
    if not isinstance(url, str) or not url:
        raise DownloadError("url must be a non-empty string")
    if len(url) > MAX_URL_LEN:
        raise DownloadError("url too long")
    if any(ord(c) < 33 or ord(c) == 127 for c in url):
        raise DownloadError("url contains whitespace or control characters")
    try:
        p = urlsplit(url)
        port = p.port
    except ValueError:
        raise DownloadError("url is malformed") from None
    # Deliberately no URL in these messages: it may carry credentials.
    if p.username is not None or p.password is not None or "@" in p.netloc:
        raise DownloadError("urls with embedded credentials are not allowed")
    if p.scheme not in policy.allowed_schemes:
        raise DownloadError(f"scheme {p.scheme!r} not allowed")
    host = (p.hostname or "").lower()
    if not host:
        raise DownloadError("url has no host")
    if host not in policy.allowed_hosts:
        raise DownloadError(f"host {host!r} is not on the allowlist")
    if port is not None and port not in policy.allowed_ports:
        raise DownloadError(f"port {port} not allowed")
    if port is None and p.scheme == "https" and 443 not in policy.allowed_ports:
        raise DownloadError("default port not allowed")
    return url


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    max_redirections = MAX_REDIRECTS

    def __init__(self, policy: Policy):
        self.policy = policy

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urljoin(req.full_url, newurl)
        validate_url(target, self.policy)  # raises DownloadError, aborting the fetch
        return super().redirect_request(req, fp, code, msg, headers, target)


def _build_opener(policy: Policy) -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()  # verifies certificates and hostnames
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),  # direct connections only
        urllib.request.HTTPSHandler(context=ctx),
        _CheckedRedirects(policy),
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def fetch(
    url: str,
    rel_dest: str,
    sha256: str,
    max_bytes: int,
    *,
    base_dir: str | Path = DATA_DIR / "raw",
    policy: Policy = DEFAULT_POLICY,
    timeout: float = 30.0,
    max_seconds: float = 600.0,
) -> Path:
    """Download `url` to `base_dir/rel_dest` and return the final path.

    Reuses an existing file whose sha256 already matches (no network call).
    """
    if not isinstance(sha256, str) or not _SHA_RE.match(sha256):
        raise DownloadError("sha256 must be 64 lowercase hex characters")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise DownloadError("max_bytes must be a positive integer")
    validate_url(url, policy)

    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    dest = safe_join(base, rel_dest)
    if dest == base.resolve() or dest.is_dir():
        raise DownloadError("destination must be a file path")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file() and sha256_file(dest) == sha256:
        log.info("reusing verified file %s", dest.name)
        return dest

    shown = _redact(url)
    req = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out:
            digest = hashlib.sha256()
            total = 0
            deadline = time.monotonic() + max_seconds
            try:
                with _build_opener(policy).open(req, timeout=timeout) as resp:
                    if resp.status != 200:
                        raise DownloadError(f"{shown}: unexpected HTTP status {resp.status}")
                    declared = resp.headers.get("Content-Length")
                    if declared is not None:
                        if not declared.isdigit() or int(declared) > max_bytes:
                            raise DownloadError(f"{shown}: Content-Length exceeds {max_bytes} bytes")
                    while chunk := resp.read1(CHUNK):  # read1: return per recv so the deadline is checked
                        total += len(chunk)
                        if total > max_bytes:
                            raise DownloadError(f"{shown}: response exceeds {max_bytes} bytes")
                        if time.monotonic() > deadline:
                            raise DownloadError(f"{shown}: exceeded {max_seconds}s total time")
                        digest.update(chunk)
                        out.write(chunk)
            except urllib.error.HTTPError as e:
                raise DownloadError(f"{shown}: HTTP {e.code}") from None
            except (urllib.error.URLError, http.client.HTTPException, OSError, ssl.SSLError) as e:
                raise DownloadError(f"{shown}: transfer failed ({type(e).__name__})") from None
        if digest.hexdigest() != sha256:
            raise DownloadError(f"{shown}: sha256 mismatch (got {digest.hexdigest()})")
        os.chmod(tmp, 0o644)
        os.replace(tmp, dest)  # atomic on the same filesystem
        log.info("downloaded %s (%d bytes)", dest.name, total)
        return dest
    finally:
        tmp.unlink(missing_ok=True)
