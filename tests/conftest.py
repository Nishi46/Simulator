"""Block real network access in the test suite: only loopback connections are allowed.

Tests that genuinely need the internet must be marked `@pytest.mark.network`.
"""
import ipaddress
import socket

import pytest

_real_connect = socket.socket.connect
_real_getaddrinfo = socket.getaddrinfo


def _is_local(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode()
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _block_network(request, monkeypatch):
    if request.node.get_closest_marker("network"):
        return

    def connect(self, address, *a, **k):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(address[0]):
            raise RuntimeError(f"network access blocked in tests: {address[0]}")
        return _real_connect(self, address, *a, **k)

    def getaddrinfo(host, *a, **k):
        if host is not None and not _is_local(host):
            raise RuntimeError(f"DNS lookup blocked in tests: {host}")
        return _real_getaddrinfo(host, *a, **k)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
