from dataclasses import FrozenInstanceError

import pytest

from jevkit.models import JEV_1_13, JEV_1_14, KNOWN, PINNED, JevModel, model_version


@pytest.mark.parametrize(("model_id", "version"), [
    ("jev-1.14.0", "1.14"),
    ("jev-1.14", "1.14"),
    ("typesafe/jev-1.14-20261014", "1.14"),
    ("~typesafe/jev-1.14", "1.14"),
    ("typesafe/jev-1.13", "1.13"),
    ("1.14", "1.14"),
])
def test_model_version_normalizes_provider_model_ids(model_id, version):
    assert model_version(model_id) == version


@pytest.mark.parametrize("model_id", [
    "jev-latest", "~typesafe/jev-latest", "static", "clef-flash", "jev-1.x",
    "typesafe/1.14", "~jev-1.14",
])
def test_model_version_rejects_aliases_and_unknown_models(model_id):
    assert model_version(model_id) is None


def test_known_models_are_immutable_and_pin_jev_114():
    assert JEV_1_14 == JevModel("1.14", "jev-1.14.0", "typesafe/jev-1.14-20261014")
    assert JEV_1_13 == JevModel("1.13", "jev-1.13.0", "typesafe/jev-1.13-20260917")
    assert PINNED is JEV_1_14 and KNOWN == (JEV_1_13, JEV_1_14)
    with pytest.raises(FrozenInstanceError):
        JEV_1_14.version = "1.15"
