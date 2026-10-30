"""Thread-safe pub/sub for live training + VM step events (SSE consumers)."""

from __future__ import annotations

import json
import queue
import threading
from typing import Any

_lock = threading.Lock()
_subscribers: list[queue.Queue[str]] = []


def publish(event: dict[str, Any]) -> None:
    payload = json.dumps(event)
    with _lock:
        dead: list[queue.Queue[str]] = []
        for q in _subscribers:
            try:
                q.put_nowait(payload)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    dead.append(q)
        for q in dead:
            _subscribers.remove(q)


def subscribe(maxsize: int = 512) -> queue.Queue[str]:
    q: queue.Queue[str] = queue.Queue(maxsize=maxsize)
    with _lock:
        _subscribers.append(q)
    return q


def unsubscribe(q: queue.Queue[str]) -> None:
    with _lock:
        if q in _subscribers:
            _subscribers.remove(q)
