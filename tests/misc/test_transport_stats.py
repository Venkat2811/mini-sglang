from __future__ import annotations

import socket
import time

from minisgl.env import ENV
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, transport_stats_snapshot


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_transport_latency_stats_capture_send_and_receive():
    old = ENV.TRANSPORT_LATENCY_STATS.value
    ENV.TRANSPORT_LATENCY_STATS.value = True
    transport_stats_snapshot(reset=True)
    port = _free_port()
    addr = f"tcp://127.0.0.1:{port}"
    push = ZmqPushQueue(addr, create=True, encoder=lambda x: x)
    pull = ZmqPullQueue(addr, create=False, decoder=lambda x: x)
    try:
        # Let sockets finish handshake before first message timing sample.
        time.sleep(0.01)
        push.put({"kind": "ping", "v": 1})
        got = pull.get()
        assert got == {"kind": "ping", "v": 1}
        stats = transport_stats_snapshot()
        assert stats["enabled"] is True
        assert int(stats["messages_sent"]) >= 1
        assert int(stats["messages_recv"]) >= 1
        assert int(stats["bytes_sent"]) > 0
        assert int(stats["bytes_recv"]) > 0
        assert float(stats["avg_pack_us"]) >= 0.0
        assert float(stats["avg_unpack_us"]) >= 0.0
    finally:
        push.stop()
        pull.stop()
        ENV.TRANSPORT_LATENCY_STATS.value = old
        transport_stats_snapshot(reset=True)
