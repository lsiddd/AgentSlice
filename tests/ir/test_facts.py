from agentslice.ir.facts import Fact


def test_fact_defaults_to_no_predecessor() -> None:
    fact = Fact(key="x", value=1, origin_event_id="e1")
    assert fact.supersedes is None


def test_fact_records_predecessor() -> None:
    fact = Fact(key="x", value=2, origin_event_id="e2", supersedes="e1")
    assert fact.supersedes == "e1"
