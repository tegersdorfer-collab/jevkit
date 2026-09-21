"""
Cache keyed on (model, state, questions). Jev is nearly deterministic (std dev 0.01
across repeats), so a hit is as good as a fresh call - and saves one. LRU with TTL;
expired entries are removed lazily on access.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from jevkit.backends import RawResponse


def cache_key(model: str | None, state: Any, questions: dict[str, dict]) -> str:
    blob = json.dumps({"m": model, "s": state, "q": questions}, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class MemoryCache:
    def __init__(self, maxsize: int = 1024, ttl_s: float = 3600.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, RawResponse]] = OrderedDict()

    def get(self, key: str) -> RawResponse | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires, value = item
        if self._clock() >= expires:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def put(self, key: str, value: RawResponse) -> None:
        self._data[key] = (self._clock() + self.ttl_s, value)
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
