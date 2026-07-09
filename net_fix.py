"""net_fix.py — workaround for networks that advertise IPv6 but blackhole it.

Some networks/routers return IPv6 addresses from DNS for a host but never
actually route IPv6 traffic. socket.create_connection() tries every address
in the order DNS returns them, so on such networks the first connection in a
process can burn the full OS TCP connect timeout on each unreachable IPv6
address before finally succeeding over IPv4 — adding tens of seconds of
latency to the first API call.
"""

import socket

_original_getaddrinfo = socket.getaddrinfo


def prefer_ipv4() -> None:
    """Make socket connections resolve IPv4 addresses only."""

    def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = _ipv4_only_getaddrinfo
