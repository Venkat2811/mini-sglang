from __future__ import annotations

from threading import Lock
from time import perf_counter_ns
from typing import Callable, Dict, Generic, TypeVar

import msgpack
import zmq
import zmq.asyncio
from minisgl.env import ENV

T = TypeVar("T")


class _TransportStats:
    def __init__(self) -> None:
        self._lock = Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.messages_sent = 0
            self.messages_recv = 0
            self.bytes_sent = 0
            self.bytes_recv = 0
            self.encode_ns = 0
            self.decode_ns = 0
            self.pack_ns = 0
            self.unpack_ns = 0
            self.send_ns = 0
            self.recv_ns = 0

    def record_send(self, *, size: int, encode_ns: int, pack_ns: int, send_ns: int) -> None:
        with self._lock:
            self.messages_sent += 1
            self.bytes_sent += size
            self.encode_ns += encode_ns
            self.pack_ns += pack_ns
            self.send_ns += send_ns

    def record_recv(self, *, size: int, recv_ns: int, unpack_ns: int, decode_ns: int) -> None:
        with self._lock:
            self.messages_recv += 1
            self.bytes_recv += size
            self.recv_ns += recv_ns
            self.unpack_ns += unpack_ns
            self.decode_ns += decode_ns

    def snapshot(self, *, reset: bool = False) -> Dict[str, int | float | bool]:
        with self._lock:
            sent = self.messages_sent
            recv = self.messages_recv
            payload = {
                "enabled": bool(ENV.TRANSPORT_LATENCY_STATS),
                "messages_sent": sent,
                "messages_recv": recv,
                "bytes_sent": self.bytes_sent,
                "bytes_recv": self.bytes_recv,
                "encode_ns": self.encode_ns,
                "decode_ns": self.decode_ns,
                "pack_ns": self.pack_ns,
                "unpack_ns": self.unpack_ns,
                "send_ns": self.send_ns,
                "recv_ns": self.recv_ns,
                "avg_encode_us": (self.encode_ns / sent / 1000.0) if sent else 0.0,
                "avg_pack_us": (self.pack_ns / sent / 1000.0) if sent else 0.0,
                "avg_send_us": (self.send_ns / sent / 1000.0) if sent else 0.0,
                "avg_recv_us": (self.recv_ns / recv / 1000.0) if recv else 0.0,
                "avg_unpack_us": (self.unpack_ns / recv / 1000.0) if recv else 0.0,
                "avg_decode_us": (self.decode_ns / recv / 1000.0) if recv else 0.0,
            }
            if reset:
                self.messages_sent = 0
                self.messages_recv = 0
                self.bytes_sent = 0
                self.bytes_recv = 0
                self.encode_ns = 0
                self.decode_ns = 0
                self.pack_ns = 0
                self.unpack_ns = 0
                self.send_ns = 0
                self.recv_ns = 0
            return payload


_TRANSPORT_STATS = _TransportStats()


def _stats_enabled() -> bool:
    return bool(ENV.TRANSPORT_LATENCY_STATS)


def transport_stats_snapshot(*, reset: bool = False) -> Dict[str, int | float | bool]:
    return _TRANSPORT_STATS.snapshot(reset=reset)


class ZmqPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put(self, obj: T):
        if _stats_enabled():
            t0 = perf_counter_ns()
            payload = self.encoder(obj)
            t1 = perf_counter_ns()
            event = msgpack.packb(payload, use_bin_type=True)
            t2 = perf_counter_ns()
            self.socket.send(event, copy=False)
            t3 = perf_counter_ns()
            _TRANSPORT_STATS.record_send(
                size=len(event),
                encode_ns=t1 - t0,
                pack_ns=t2 - t1,
                send_ns=t3 - t2,
            )
            return
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqAsyncPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    async def put(self, obj: T):
        if _stats_enabled():
            t0 = perf_counter_ns()
            payload = self.encoder(obj)
            t1 = perf_counter_ns()
            event = msgpack.packb(payload, use_bin_type=True)
            t2 = perf_counter_ns()
            await self.socket.send(event, copy=False)
            t3 = perf_counter_ns()
            _TRANSPORT_STATS.record_send(
                size=len(event),
                encode_ns=t1 - t0,
                pack_ns=t2 - t1,
                send_ns=t3 - t2,
            )
            return
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        await self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    def get(self) -> T:
        if _stats_enabled():
            t0 = perf_counter_ns()
            event = self.socket.recv()
            t1 = perf_counter_ns()
            payload = msgpack.unpackb(event, raw=False)
            t2 = perf_counter_ns()
            out = self.decoder(payload)
            t3 = perf_counter_ns()
            _TRANSPORT_STATS.record_recv(
                size=len(event),
                recv_ns=t1 - t0,
                unpack_ns=t2 - t1,
                decode_ns=t3 - t2,
            )
            return out
        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def get_raw(self) -> bytes:
        event = self.socket.recv()
        if _stats_enabled():
            _TRANSPORT_STATS.record_recv(size=len(event), recv_ns=0, unpack_ns=0, decode_ns=0)
        return event

    def decode(self, raw: bytes) -> T:
        if _stats_enabled():
            t0 = perf_counter_ns()
            payload = msgpack.unpackb(raw, raw=False)
            t1 = perf_counter_ns()
            out = self.decoder(payload)
            t2 = perf_counter_ns()
            _TRANSPORT_STATS.record_recv(
                size=len(raw),
                recv_ns=0,
                unpack_ns=t1 - t0,
                decode_ns=t2 - t1,
            )
            return out
        return self.decoder(msgpack.unpackb(raw, raw=False))

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqAsyncPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    async def get(self) -> T:
        if _stats_enabled():
            t0 = perf_counter_ns()
            event = await self.socket.recv()
            t1 = perf_counter_ns()
            payload = msgpack.unpackb(event, raw=False)
            t2 = perf_counter_ns()
            out = self.decoder(payload)
            t3 = perf_counter_ns()
            _TRANSPORT_STATS.record_recv(
                size=len(event),
                recv_ns=t1 - t0,
                unpack_ns=t2 - t1,
                decode_ns=t3 - t2,
            )
            return out
        event = await self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqPubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put_raw(self, raw: bytes):
        if _stats_enabled():
            t0 = perf_counter_ns()
            self.socket.send(raw, copy=False)
            t1 = perf_counter_ns()
            _TRANSPORT_STATS.record_send(size=len(raw), encode_ns=0, pack_ns=0, send_ns=t1 - t0)
            return
        self.socket.send(raw, copy=False)

    def put(self, obj: T):
        if _stats_enabled():
            t0 = perf_counter_ns()
            payload = self.encoder(obj)
            t1 = perf_counter_ns()
            event = msgpack.packb(payload, use_bin_type=True)
            t2 = perf_counter_ns()
            self.socket.send(event, copy=False)
            t3 = perf_counter_ns()
            _TRANSPORT_STATS.record_send(
                size=len(event),
                encode_ns=t1 - t0,
                pack_ns=t2 - t1,
                send_ns=t3 - t2,
            )
            return
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqSubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.decoder = decoder

    def get(self) -> T:
        if _stats_enabled():
            t0 = perf_counter_ns()
            event = self.socket.recv()
            t1 = perf_counter_ns()
            payload = msgpack.unpackb(event, raw=False)
            t2 = perf_counter_ns()
            out = self.decoder(payload)
            t3 = perf_counter_ns()
            _TRANSPORT_STATS.record_recv(
                size=len(event),
                recv_ns=t1 - t0,
                unpack_ns=t2 - t1,
                decode_ns=t3 - t2,
            )
            return out
        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()
