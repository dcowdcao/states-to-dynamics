"""Force IPv4 for HuggingFace downloads on clusters (FASRC IPv6 flakiness)."""

# def force_ipv4() -> None:
#     # monkeypatch socket.getaddrinfo to prefer AF_INET before any HF download.
