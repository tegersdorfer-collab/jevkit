import asyncio

import pytest

from jevkit.backends import StaticBackend
from jevkit.client import Client, JevUnavailable
from jevkit.gate import Band, Bands, Severity
from jevkit.questions import Noul
from jevkit.registry import DecisionSpec, Privacy, Registry, Router, Verdict


def _reg():
    r = Registry()
    r.register(DecisionSpec("addressed",
                            Noul("Is the user talking to the assistant?"),
                            severity=Severity.READ))
    r.register(DecisionSpec("delete_ok", Noul("Delete?"), severity=Severity.DESTRUCTIVE,
                            calibrated_model="typesafe/jev-1.13"))
    r.register(DecisionSpec("fact", Noul("Fact supported?"), privacy=Privacy.LOCAL,
                            bands=Bands(0.5, 0.0)))
    return r


def test_registry_basics():
    r = _reg()
    assert "addressed" in r and "nope" not in r and [s.id for s in r] == ["addressed", "delete_ok", "fact"]
    assert r.get("fact").effective_bands == Bands(0.5, 0.0)
    assert r.get("addressed").effective_bands == Bands(0.4, 0.2)
    assert list(r.questions(["fact", "addressed"])) == ["fact", "addressed"]
    with pytest.raises(ValueError):
        r.register(DecisionSpec("fact", Noul("dup")))
    with pytest.raises(KeyError):
        r.get("nope")


def test_router_teilt_nach_privacy_und_baendert():
    cloud = StaticBackend({"addressed": {"type": "noul", "noul": 0.95},
                           "delete_ok": {"type": "noul", "noul": 0.8}},
                          model="typesafe/jev-1.13.0")
    local = StaticBackend({"fact": {"type": "noul", "noul": 0.9}}, model="prompt:qwen")
    router = Router(_reg(), cloud=Client(cloud), local=Client(local))
    v = asyncio.run(router.decide({"t": "x"}, ["addressed", "delete_ok", "fact"]))
    assert set(v) == {"addressed", "delete_ok", "fact"}
    assert v["addressed"] == Verdict("addressed", v["addressed"].answer, Band.ACT,
                                     "typesafe/jev-1.13.0", True)
    assert v["delete_ok"].band is Band.CONFIRM          # conf 0.6 < 0.8 (DESTRUCTIVE)
    assert v["fact"].band is Band.ACT and v["fact"].model == "prompt:qwen"
    assert set(cloud.calls[0][1]) == {"addressed", "delete_ok"} and set(local.calls[0][1]) == {"fact"}


def test_router_unkalibriertes_modell_wird_demotet():
    cloud = StaticBackend({"delete_ok": {"type": "noul", "noul": 0.99}}, model="typesafe/jev-2.0")
    v = asyncio.run(Router(_reg(), cloud=Client(cloud)).decide("x", ["delete_ok"]))
    assert v["delete_ok"].calibrated is False and v["delete_ok"].band is Band.CONFIRM


def test_router_faellt_von_cloud_auf_local():
    cloud = Client(StaticBackend({}))                      # liefert keine Antworten → JevUnavailable
    local = StaticBackend({"addressed": {"type": "noul", "noul": 0.9}}, model="prompt:local")
    v = asyncio.run(Router(_reg(), cloud=cloud, local=Client(local)).decide("x", ["addressed"]))
    assert v["addressed"].model == "prompt:local" and v["addressed"].band is Band.ACT


def test_router_ohne_passendes_backend():
    with pytest.raises(JevUnavailable):
        asyncio.run(Router(_reg(),
                           cloud=Client(StaticBackend({"addressed": {"type": "noul",
                                                                      "noul": 0.9}}))
                          ).decide("x", ["fact"]))
    with pytest.raises(JevUnavailable):
        asyncio.run(Router(_reg()).decide("x", ["addressed"]))
    with pytest.raises(KeyError):
        asyncio.run(Router(_reg(), cloud=Client(StaticBackend({}))).decide("x",
                                                                            ["nope"]))
