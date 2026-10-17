import json
from collections import Counter
from pathlib import Path


DATASET = Path(__file__).parents[1] / "bench" / "data" / "guard_eval.json"


def test_routed_cases_are_paired_balanced_and_have_explicit_sources():
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in cases}
    assert len(by_id) == len(cases)

    routed = [case for case in cases if case.get("category") == "routed_in_content"]
    utterances = [case for case in cases if case.get("category") == "routed_user_utterance"]
    legacy = [case for case in cases if case.get("category") not in {
        "routed_in_content", "routed_user_utterance", "source_spoof", "source_mention",
    }]
    assert all("source" not in case for case in legacy)
    assert len(routed) == len(utterances) == 40
    assert Counter((case["split"], case["lang"]) for case in routed) == {
        ("dev", "de"): 10, ("dev", "en"): 10,
        ("test", "de"): 10, ("test", "en"): 10,
    }
    assert Counter((case["split"], case["lang"]) for case in utterances) == {
        ("dev", "de"): 10, ("dev", "en"): 10,
        ("test", "de"): 10, ("test", "en"): 10,
    }

    for content in routed:
        assert content["injection"] is True and content["self_claim"] is False
        assert content["source"] == "third_party_content"
        assert content["pair"] in by_id
        user = by_id[content["pair"]]
        assert user["category"] == "routed_user_utterance"
        assert user["source"] == "user_utterance"
        assert user["injection"] is False and user["self_claim"] is False
        assert user["split"] == content["split"] and user["lang"] == content["lang"]
        assert user["pair"] == content["id"]
        assert user["text"] in content["text"]


def test_own_data_requests_have_balanced_new_dev_cases_without_source_labels():
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    own_data = [case for case in cases if case.get("category") == "own_data_request"]
    dev = [case for case in own_data if case["split"] == "dev"]
    test = [case for case in own_data if case["split"] == "test"]

    assert len(test) == 22
    assert {case["id"] for case in test} == {f"own-data-{i:02d}" for i in range(22)}
    assert len(dev) == 16
    assert {case["id"] for case in dev} == {f"own-data-{i:02d}" for i in range(22, 38)}
    assert Counter(case["lang"] for case in dev) == {"de": 8, "en": 8}
    assert all(case["injection"] is False and case["self_claim"] is False for case in dev)
    assert all("source" not in case for case in dev)


def test_source_spoof_cases_are_balanced_and_mix_labeled_and_unlabeled_sources():
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    spoof = [case for case in cases if case.get("category") == "source_spoof"]
    mention = [case for case in cases if case.get("category") == "source_mention"]
    assert len(spoof) == 16 and all(case["injection"] for case in spoof)
    assert Counter((case["split"], case["lang"]) for case in spoof) == {
        ("dev", "de"): 4, ("dev", "en"): 4, ("test", "de"): 4, ("test", "en"): 4,
    }
    assert Counter(case.get("source") for case in spoof) == {"third_party_content": 8, None: 8}
    assert len(mention) == 4 and not any(case["injection"] for case in mention)
    assert {case["source"] for case in mention} == {"user_utterance"}
    assert Counter(case["split"] for case in mention) == {"dev": 2, "test": 2}
