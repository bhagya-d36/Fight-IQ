"""net_fix.py — workaround for networks that advertise IPv6 but blackhole it.

Some networks/routers return IPv6 addresses from DNS for a host but never
actually route IPv6 traffic. socket.create_connection() tries every address
in the order DNS returns them, so on such networks the first connection in a
process can burn the full OS TCP connect timeout on each unreachable IPv6
address before finally succeeding over IPv4 — adding tens of seconds of
latency to the first API call.

The patch changes DNS resolution for the WHOLE process (it would break real
IPv6 connectivity), so it is opt-in: set FORCE_IPV4=1 in .env if your first
API call hangs for ~20 seconds.
"""

import os
import socket

_original_getaddrinfo = socket.getaddrinfo


def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


def prefer_ipv4(force: bool | None = None) -> None:
    """Make socket connections resolve IPv4 addresses only.

    No-op unless FORCE_IPV4 is set in the environment (or force=True).
    Reads os.environ directly so this module stays dependency-free; callers
    import config first, which loads .env before this runs.
    """
    if force is None:
        force = os.environ.get("FORCE_IPV4", "").strip().lower() in ("1", "true", "yes", "on")
    if force:
        socket.getaddrinfo = _ipv4_only_getaddrinfo
