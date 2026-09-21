from jevkit.backends import RawResponse
from jevkit.cache import MemoryCache, cache_key


def test_cache_key_ist_stabil_und_ordnungsunabhaengig():
    a = cache_key("m", {"x": 1, "y": [1, 2]}, {"q": {"type": "noul", "instructions": "?"}})
    b = cache_key("m", {"y": [1, 2], "x": 1}, {"q": {"instructions": "?", "type": "noul"}})
    assert a == b and len(a) == 64
    assert cache_key("m2", {"x": 1}, {}) != cache_key("m", {"x": 1}, {})
    assert cache_key("m", "ä", {}) == cache_key("m", "ä", {})


def test_memory_cache_lru_und_ttl():
    t = {"now": 0.0}
    c = MemoryCache(maxsize=2, ttl_s=10.0, clock=lambda: t["now"])
    r1, r2, r3 = (RawResponse(f"m{i}", {}, {}) for i in range(3))
    c.put("a", r1)
    c.put("b", r2)
    assert c.get("a") is r1 and len(c) == 2
    c.put("c", r3)                       # verdrängt "b" (a wurde zuletzt gelesen)
    assert c.get("b") is None and c.get("a") is r1 and c.get("c") is r3
    t["now"] = 11.0
    assert c.get("a") is None and len(c) == 1  # abgelaufen → entfernt; "c" wurde bei 0 gesetzt, auch weg?
    c.clear()
    assert len(c) == 0
