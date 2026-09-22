from __future__ import annotations

import pytest
from p3fcl.attacks.base import Attack, ThreatModel, UnspecifiedThreatModelError


def test_threat_model_validates_fields():
    tm = ThreatModel(
        who="hbc_server", active=False, view="full", auxiliary="shadow_training",
        target="membership", survives_secure_agg=True,
    )
    assert tm.who == "hbc_server"


@pytest.mark.parametrize(
    "field_,value",
    [("who", "attacker"), ("view", "sometimes"), ("auxiliary", "telepathy"), ("target", "mind-reading")],
)
def test_threat_model_rejects_invalid_values(field_, value):
    kwargs = dict(
        who="hbc_server", active=False, view="full", auxiliary="none",
        target="membership", survives_secure_agg=True,
    )
    kwargs[field_] = value
    with pytest.raises(ValueError):
        ThreatModel(**kwargs)


def test_attack_without_threat_model_refuses_to_instantiate():
    class NoThreatModelAttack(Attack):
        pass

    with pytest.raises(UnspecifiedThreatModelError):
        NoThreatModelAttack({})


def test_attack_with_threat_model_instantiates():
    class FineAttack(Attack):
        threat_model = ThreatModel(
            who="participating_client", active=False, view="task", auxiliary="public_data",
            target="reconstruction", survives_secure_agg=False,
        )

    a = FineAttack({})
    with pytest.raises(NotImplementedError):
        a.run(ledger=None)
