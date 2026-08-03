"""Force IPv4 for network calls.

Python's socket prefers IPv6, which hangs on networks (some clusters) that don't
route it — unlike curl's Happy-Eyeballs fallback. Call `force_ipv4()` before any
HuggingFace download.
"""

import socket

_orig_getaddrinfo = socket.getaddrinfo


def force_ipv4() -> None:
    def ipv4_only(*args, **kwargs):
        results = _orig_getaddrinfo(*args, **kwargs)
        ipv4 = [r for r in results if r[0] == socket.AF_INET]
        return ipv4 if ipv4 else results
    socket.getaddrinfo = ipv4_only
