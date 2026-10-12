import hashlib
import http.server
import logging
import os
import threading
import time

import pytest

from data import download
from data.download import DEFAULT_POLICY, DownloadError, Policy, fetch, validate_url
from data.paths import UnsafePathError

BODY = b"respondent data " * 100
SHA = hashlib.sha256(BODY).hexdigest()


class Server:
    """Local HTTP server; routes map path -> handler(h)."""

    def __init__(self, routes):
        self.hits = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.hits.append(self.path)
                routes[self.path.split("?")[0]](self)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.block_on_close = False  # don't wait for deliberately stalled handlers
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.policy = Policy(("127.0.0.1",), ("http",), (self.port,))

    def url(self, path="/f"):
        return f"http://127.0.0.1:{self.port}{path}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def ok(body=BODY):
    def h(r):
        r.send_response(200)
        r.send_header("Content-Length", str(len(body)))
        r.end_headers()
        r.wfile.write(body)
    return h


@pytest.fixture
def serve():
    servers = []

    def make(routes):
        s = Server(routes)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


def go(s, tmp_path, path="/f", sha=SHA, max_bytes=10_000, rel="out/f.bin", **kw):
    return fetch(s.url(path), rel, sha, max_bytes, base_dir=tmp_path, policy=s.policy, **kw)


def leftovers(tmp_path):
    return [p for p in tmp_path.rglob("*") if p.is_file() and p.name.endswith(".part")]


# ---- URL validation -------------------------------------------------------

def test_default_policy_is_https_only_allowlisted():
    assert validate_url("https://gss.norc.org/x.zip")
    assert validate_url("https://GSS.NORC.ORG/x.zip")
    for bad in [
        "http://gss.norc.org/x", "ftp://gss.norc.org/x", "file:///etc/passwd",
        "https://evil.example/x", "https://gss.norc.org.evil.example/x",
        "https://evilgss.norc.org/x", "https://gss.norc.org:8443/x",
        "https://gss.norc.org./x", "https:///x", "//gss.norc.org/x", "gss.norc.org/x",
        "https://127.0.0.1/x", "https://[::1]/x", "", None, 5,
        "https://gss.norc.org/x y", "https://gss.norc.org/x\n", "https://gss.norc.org/\x00",
        "https://gss.norc.org/" + "a" * 3000,
    ]:
        with pytest.raises(DownloadError):
            validate_url(bad)


def test_embedded_credentials_rejected_without_leaking():
    for url in ["https://user:hunter2@gss.norc.org/x", "https://hunter2@gss.norc.org/x",
                "https://gss.norc.org@evil.example/x", "https://evil.example@gss.norc.org/x"]:
        with pytest.raises(DownloadError) as e:
            validate_url(url)
        assert "hunter2" not in str(e.value) and "evil" not in str(e.value)


def test_tls_context_verifies_certificates():
    opener = download._build_opener(DEFAULT_POLICY)
    https = [h for h in opener.handlers if h.__class__.__name__ == "HTTPSHandler"][0]
    assert https._context.verify_mode.name == "CERT_REQUIRED"
    assert https._context.check_hostname


# ---- fetch: happy path and parameter checks -------------------------------

def test_fetch_success(serve, tmp_path):
    s = serve({"/f": ok()})
    p = go(s, tmp_path)
    assert p == (tmp_path / "out/f.bin").resolve()
    assert p.read_bytes() == BODY
    assert not leftovers(tmp_path)
    assert oct(p.stat().st_mode & 0o777) == "0o644"


def test_size_exactly_at_limit_ok_one_under_fails(serve, tmp_path):
    s = serve({"/f": ok()})
    assert go(s, tmp_path, max_bytes=len(BODY)).exists()
    with pytest.raises(DownloadError):
        go(s, tmp_path, rel="other.bin", max_bytes=len(BODY) - 1)
    assert not (tmp_path / "other.bin").exists()


@pytest.mark.parametrize("sha", ["", "abc", "A" * 64, "g" * 64, None, 5, SHA + "0"])
def test_bad_sha_param(serve, tmp_path, sha):
    s = serve({"/f": ok()})
    with pytest.raises(DownloadError, match="sha256"):
        go(s, tmp_path, sha=sha)
    assert s.hits == []


@pytest.mark.parametrize("mb", [0, -1, True, 1.5, "10", None])
def test_bad_max_bytes_param(serve, tmp_path, mb):
    s = serve({"/f": ok()})
    with pytest.raises(DownloadError, match="max_bytes"):
        go(s, tmp_path, max_bytes=mb)
    assert s.hits == []


def test_default_policy_refuses_local_http(serve, tmp_path):
    s = serve({"/f": ok()})
    with pytest.raises(DownloadError):
        fetch(s.url(), "f.bin", SHA, 10_000, base_dir=tmp_path)
    assert s.hits == []


# ---- size limits -----------------------------------------------------------

def test_declared_content_length_too_large(serve, tmp_path):
    def h(r):
        r.send_response(200)
        r.send_header("Content-Length", "999999999")
        r.end_headers()
        r.wfile.write(b"x" * 10)
    s = serve({"/f": h})
    with pytest.raises(DownloadError, match="Content-Length"):
        go(s, tmp_path)
    assert not (tmp_path / "out/f.bin").exists() and not leftovers(tmp_path)


def test_streamed_body_larger_than_limit_without_content_length(serve, tmp_path):
    def h(r):  # HTTP/1.0 close-delimited body, no Content-Length
        r.send_response(200)
        r.end_headers()
        r.wfile.write(b"x" * 500_000)
    s = serve({"/f": h})
    with pytest.raises(DownloadError, match="exceeds"):
        go(s, tmp_path, max_bytes=100_000)
    assert not (tmp_path / "out/f.bin").exists() and not leftovers(tmp_path)


def test_malformed_content_length_rejected(serve, tmp_path):
    def h(r):
        r.send_response(200)
        r.send_header("Content-Length", "-5")
        r.end_headers()
    s = serve({"/f": h})
    with pytest.raises(DownloadError):
        go(s, tmp_path)


# ---- integrity and atomicity ----------------------------------------------

def test_checksum_mismatch_leaves_nothing(serve, tmp_path):
    s = serve({"/f": ok(b"tampered")})
    with pytest.raises(DownloadError, match="sha256 mismatch"):
        go(s, tmp_path)
    assert not (tmp_path / "out/f.bin").exists() and not leftovers(tmp_path)


def test_truncated_response_keeps_previous_file(serve, tmp_path):
    def h(r):
        r.send_response(200)
        r.send_header("Content-Length", str(len(BODY)))
        r.end_headers()
        r.wfile.write(BODY[:50])  # then close early
    s = serve({"/f": h})
    dest = tmp_path / "out/f.bin"
    dest.parent.mkdir()
    dest.write_bytes(b"old good file")
    with pytest.raises(DownloadError):
        go(s, tmp_path)
    assert dest.read_bytes() == b"old good file" and not leftovers(tmp_path)


def test_failed_replace_cleans_temp(serve, tmp_path, monkeypatch):
    s = serve({"/f": ok()})

    def boom(*a):
        raise OSError("disk full")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        go(s, tmp_path)
    assert not leftovers(tmp_path) and not (tmp_path / "out/f.bin").exists()


def test_existing_matching_file_reused_without_network(serve, tmp_path):
    s = serve({"/f": ok()})
    dest = tmp_path / "out/f.bin"
    dest.parent.mkdir()
    dest.write_bytes(BODY)
    assert go(s, tmp_path).read_bytes() == BODY
    assert s.hits == []


def test_existing_wrong_file_replaced_only_after_verification(serve, tmp_path):
    dest = tmp_path / "out/f.bin"
    dest.parent.mkdir()
    dest.write_bytes(b"stale")
    bad = serve({"/f": ok(b"tampered")})
    with pytest.raises(DownloadError):
        go(bad, tmp_path)
    assert dest.read_bytes() == b"stale"
    good = serve({"/f": ok()})
    go(good, tmp_path)
    assert dest.read_bytes() == BODY


# ---- HTTP behaviour --------------------------------------------------------

@pytest.mark.parametrize("code", [404, 403, 500, 204])
def test_non_200_rejected(serve, tmp_path, code):
    def h(r):
        r.send_response(code)
        r.send_header("Content-Length", "6")
        r.end_headers()
        if code != 204:
            r.wfile.write(b"secret")
    s = serve({"/f": h})
    with pytest.raises(DownloadError) as e:
        go(s, tmp_path)
    assert "secret" not in str(e.value)
    assert not (tmp_path / "out/f.bin").exists() and not leftovers(tmp_path)


def test_redirect_within_policy_followed(serve, tmp_path):
    def redir(r):
        r.send_response(302)
        r.send_header("Location", "/real")
        r.end_headers()
    s = serve({"/f": redir, "/real": ok()})
    assert go(s, tmp_path).read_bytes() == BODY


@pytest.mark.parametrize("location", [
    "http://evil.example/x",               # host not allowlisted
    "https://127.0.0.1/x",                 # scheme not allowed by this policy
    "http://user:hunter2@127.0.0.1/x",     # credentials
    "ftp://127.0.0.1/x",
    "file:///etc/passwd",
])
def test_redirect_to_disallowed_target_rejected(serve, tmp_path, location):
    def redir(r):
        r.send_response(302)
        r.send_header("Location", location)
        r.end_headers()
    s = serve({"/f": redir})
    with pytest.raises(DownloadError) as e:
        go(s, tmp_path)
    assert "hunter2" not in str(e.value)
    assert not (tmp_path / "out/f.bin").exists() and not leftovers(tmp_path)


def test_redirect_loop_aborts(serve, tmp_path):
    def loop(r):
        r.send_response(302)
        r.send_header("Location", "/f")
        r.end_headers()
    s = serve({"/f": loop})
    with pytest.raises(DownloadError):
        go(s, tmp_path)
    assert len(s.hits) <= download.MAX_REDIRECTS + 2


def test_stalled_server_times_out(serve, tmp_path):
    def stall(r):
        r.send_response(200)
        r.send_header("Content-Length", "100")
        r.end_headers()
        time.sleep(3)
    s = serve({"/f": stall})
    t0 = time.monotonic()
    with pytest.raises(DownloadError):
        go(s, tmp_path, timeout=0.5)
    assert time.monotonic() - t0 < 2.5 and not leftovers(tmp_path)


def test_slow_trickle_hits_total_deadline(serve, tmp_path):
    def trickle(r):
        r.send_response(200)
        r.end_headers()
        try:
            for _ in range(50):
                r.wfile.write(b"x" * 10)
                r.wfile.flush()
                time.sleep(0.1)
        except OSError:
            pass
    s = serve({"/f": trickle})
    with pytest.raises(DownloadError):
        go(s, tmp_path, timeout=2, max_seconds=0.3)
    assert not leftovers(tmp_path)


def test_connection_refused(tmp_path):
    p = Policy(("127.0.0.1",), ("http",), (1,))
    with pytest.raises(DownloadError):
        fetch("http://127.0.0.1:1/f", "f.bin", SHA, 100, base_dir=tmp_path, policy=p)
    assert not leftovers(tmp_path)


# ---- destination safety ----------------------------------------------------

@pytest.mark.parametrize("rel", ["../escape.bin", "/abs.bin", "a/../../x", "a\\b"])
def test_destination_traversal_rejected(serve, tmp_path, rel):
    s = serve({"/f": ok()})
    base = tmp_path / "base"
    with pytest.raises(UnsafePathError):
        fetch(s.url(), rel, SHA, 10_000, base_dir=base, policy=s.policy)
    assert s.hits == []


def test_destination_symlink_outside_rejected(serve, tmp_path):
    s = serve({"/f": ok()})
    base, outside = tmp_path / "base", tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    os.symlink(outside, base / "link")
    with pytest.raises(UnsafePathError):
        fetch(s.url(), "link/f.bin", SHA, 10_000, base_dir=base, policy=s.policy)
    assert list(outside.iterdir()) == []


def test_destination_directory_rejected(serve, tmp_path):
    s = serve({"/f": ok()})
    (tmp_path / "d").mkdir()
    with pytest.raises(DownloadError, match="file path"):
        fetch(s.url(), "d", SHA, 10_000, base_dir=tmp_path, policy=s.policy)


# ---- no secret leakage -----------------------------------------------------

def test_query_string_redacted_from_errors_and_logs(serve, tmp_path, caplog):
    def missing(r):
        r.send_response(404)
        r.send_header("Content-Length", "0")
        r.end_headers()
    s = serve({"/f": missing})
    caplog.set_level(logging.DEBUG)
    with pytest.raises(DownloadError) as e:
        fetch(s.url("/f?token=abc123"), "f.bin", SHA, 100, base_dir=tmp_path, policy=s.policy)
    assert "abc123" not in str(e.value)
    assert "abc123" not in caplog.text
