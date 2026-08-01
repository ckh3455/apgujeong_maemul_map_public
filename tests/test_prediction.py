import pandas as pd

from data_loader import _parse_service_account
from prediction import floor_band, floor_from_ho, normalize_area, price_eok


def test_floor_from_ho():
    assert floor_from_ho("1203호") == 12
    assert floor_from_ho("101") == 1
    assert floor_from_ho(1502) == 15


def test_floor_band_uses_relative_floor():
    assert floor_band(10, 12) == "high"
    assert floor_band(10, 17) == "middle"
    assert floor_band(1, 17) == "first"


def test_normalize_and_price():
    assert normalize_area("03구역") == "3"
    assert price_eok("64억") == 64
    assert price_eok("640,000") == 64


def test_service_account_private_key_newlines():
    raw = '{"type":"service_account","private_key":"-----BEGIN PRIVATE KEY-----\nABC\n-----END PRIVATE KEY-----\n","client_email":"test@example.com","token_uri":"https://oauth2.googleapis.com/token"}'
    parsed = _parse_service_account(raw)
    assert parsed["client_email"] == "test@example.com"
    assert "\nABC\n" in parsed["private_key"]
