import pandas as pd
import pytest
from features import build_model_frame


def _txs(**overrides):
    row = {"transaction_id": 1, "account_id": 101, "amount_usd": 100, "failed_logins_24h": 0}
    row.update(overrides)
    return pd.DataFrame([row])


def _accounts(**overrides):
    row = {"account_id": 101, "prior_chargebacks": 0}
    row.update(overrides)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# is_large_amount
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("amount,expected", [
    (999,  0),
    (1000, 1),
    (5000, 1),
])
def test_is_large_amount(amount, expected):
    result = build_model_frame(_txs(amount_usd=amount), _accounts())
    assert result["is_large_amount"].iloc[0] == expected


# ---------------------------------------------------------------------------
# login_pressure categories
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("logins,expected", [
    (0, "none"),
    (1, "low"),
    (2, "low"),
    (3, "high"),
    (10, "high"),
])
def test_login_pressure(logins, expected):
    result = build_model_frame(_txs(failed_logins_24h=logins), _accounts())
    assert result["login_pressure"].iloc[0] == expected


# ---------------------------------------------------------------------------
# Account merge
# ---------------------------------------------------------------------------

def test_account_fields_merged():
    """prior_chargebacks from accounts.csv must be present after merge."""
    result = build_model_frame(_txs(), _accounts(prior_chargebacks=3))
    assert result["prior_chargebacks"].iloc[0] == 3


def test_unknown_account_produces_null_chargeback():
    """Transaction with no matching account should produce NaN for account fields."""
    txs = pd.DataFrame([{"transaction_id": 1, "account_id": 999, "amount_usd": 100, "failed_logins_24h": 0}])
    result = build_model_frame(txs, _accounts())
    assert pd.isna(result["prior_chargebacks"].iloc[0])
