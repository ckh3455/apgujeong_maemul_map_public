import pandas as pd

from data_loader import _frame_from_values, _parse_service_account
from prediction import complex_family, floor_band, floor_from_ho, normalize_area, price_eok


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


def test_hyundai_integrated_complex_aliases():
    assert complex_family("현대1,2차") == "현대1,2"
    assert complex_family("현대1차(12,13,21,22,31,32,33동)") == "현대1,2"
    assert complex_family("현대2차") == "현대1,2"
    assert complex_family("현대6,7차") == "현대6,7"
    assert complex_family("현대7차") == "현대6,7"
    assert complex_family("현대12차") == "현대12"


def test_duplicate_and_blank_sheet_headers():
    frame = _frame_from_values([
        ["날짜", "단지", "", "", "가격", "가격"],
        ["2026.06.04", "신현대", "", "", "61", "보조값"],
    ])
    assert list(frame.columns) == ["날짜", "단지", "__blank_3", "__blank_4", "가격", "가격__2"]
    assert frame.loc[0, "가격"] == "61"
