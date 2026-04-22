import pytest
from risk_rules import label_risk, score_transaction


def _base_tx(**overrides):
    """All-zero baseline: every signal in its lowest tier, total score = 0."""
    tx = {
        "device_risk_score": 10,
        "is_international": 0,
        "amount_usd": 100,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    tx.update(overrides)
    return tx


# ---------------------------------------------------------------------------
# label_risk — exact boundary values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score,expected", [
    (0,   "low"),
    (29,  "low"),
    (30,  "medium"),
    (59,  "medium"),
    (60,  "high"),
    (100, "high"),
])
def test_label_risk_boundaries(score, expected):
    assert label_risk(score) == expected


# ---------------------------------------------------------------------------
# score_transaction — each signal tested in isolation with exact point values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device_score,expected_points", [
    (0,   0),
    (39,  0),
    (40,  10),
    (69,  10),
    (70,  25),
    (100, 25),
])
def test_device_risk_score_tiers(device_score, expected_points):
    assert score_transaction(_base_tx(device_risk_score=device_score)) == expected_points


@pytest.mark.parametrize("is_intl,expected_points", [
    (0, 0),
    (1, 15),
])
def test_international_score(is_intl, expected_points):
    assert score_transaction(_base_tx(is_international=is_intl)) == expected_points


@pytest.mark.parametrize("amount,expected_points", [
    (0,    0),
    (499,  0),
    (500,  10),
    (999,  10),
    (1000, 25),
    (5000, 25),
])
def test_amount_tiers(amount, expected_points):
    assert score_transaction(_base_tx(amount_usd=amount)) == expected_points


@pytest.mark.parametrize("velocity,expected_points", [
    (1, 0),
    (2, 0),
    (3, 5),
    (5, 5),
    (6, 20),
    (20, 20),
])
def test_velocity_tiers(velocity, expected_points):
    assert score_transaction(_base_tx(velocity_24h=velocity)) == expected_points


@pytest.mark.parametrize("logins,expected_points", [
    (0, 0),
    (1, 0),
    (2, 10),
    (4, 10),
    (5, 20),
    (10, 20),
])
def test_failed_logins_tiers(logins, expected_points):
    assert score_transaction(_base_tx(failed_logins_24h=logins)) == expected_points


@pytest.mark.parametrize("chargebacks,expected_points", [
    (0, 0),
    (1, 5),
    (2, 20),
    (5, 20),
])
def test_prior_chargeback_tiers(chargebacks, expected_points):
    assert score_transaction(_base_tx(prior_chargebacks=chargebacks)) == expected_points


# ---------------------------------------------------------------------------
# score_transaction — score clamping
# ---------------------------------------------------------------------------

def test_score_floor_is_zero():
    assert score_transaction(_base_tx()) == 0


def test_score_ceiling_is_100():
    # All signals at max would sum to 125 without clamping
    tx = _base_tx(
        device_risk_score=85,
        is_international=1,
        amount_usd=1500,
        velocity_24h=8,
        failed_logins_24h=6,
        prior_chargebacks=3,
    )
    assert score_transaction(tx) == 100


# ---------------------------------------------------------------------------
# End-to-end classification — known fraud and clean profiles
# ---------------------------------------------------------------------------

def test_textbook_fraud_scores_high():
    """High-risk device, international, burst velocity, login failures, chargeback history."""
    fraud = _base_tx(
        device_risk_score=85,
        is_international=1,
        amount_usd=1400,
        velocity_24h=8,
        failed_logins_24h=7,
        prior_chargebacks=2,
    )
    assert label_risk(score_transaction(fraud)) == "high"


def test_clean_transaction_scores_low():
    """Domestic, low-risk device, single transaction, no history."""
    assert label_risk(score_transaction(_base_tx())) == "low"


def test_international_high_velocity_no_history_scores_medium():
    """International + high velocity but no other risk signals should land medium, not high."""
    tx = _base_tx(is_international=1, velocity_24h=6)
    score = score_transaction(tx)
    assert label_risk(score) == "medium"
