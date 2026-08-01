import json
import os
import re

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials


LISTING_TAB = "매매물건 목록"
TRADE_TAB = "거래내역"

LISTING_COLUMNS = ["상태", "구역", "단지명", "동", "평형", "평형대", "대지지분", "층수", "가격"]
TRADE_COLUMNS = [
    "구역", "단지", "단지명", "단지명(단지)", "평형", "평형대", "전용면적",
    "날짜", "거래일", "계약일", "일자", "거래일자", "계약년월", "계약일자",
    "가격", "거래가격", "거래가", "실거래가", "금액", "거래금액",
    "동", "호", "층", "해제여부", "거래유형", "비고",
]


def _sheet_id(value: str) -> str:
    match = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", str(value))
    return match.group(1) if match else str(value).strip()


def _service_account() -> dict:
    if "GCP_SERVICE_ACCOUNT_JSON" in st.secrets:
        value = st.secrets["GCP_SERVICE_ACCOUNT_JSON"]
        if isinstance(value, dict):
            return dict(value)
        return json.loads(str(value).replace("\\n", "\n"))
    env = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if env:
        return json.loads(env)
    raise RuntimeError("GCP_SERVICE_ACCOUNT_JSON이 없습니다.")


def _secret(name: str, default=None):
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)


def _read_tab(book, tab_name: str, allow: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(book.worksheet(tab_name).get_all_records())
    frame.columns = [str(c).replace("\n", "").strip() for c in frame.columns]
    return frame[[c for c in allow if c in frame.columns]].copy()


@st.cache_data(ttl=600)
def load_data():
    listing_sheet = _secret("SPREADSHEET_ID")
    if not listing_sheet:
        raise RuntimeError("SPREADSHEET_ID가 없습니다.")

    info = _service_account()
    credentials = Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly", "https://www.googleapis.com/auth/drive.readonly"],
    )
    client = gspread.authorize(credentials)
    listing_book = client.open_by_key(_sheet_id(listing_sheet))
    listings = _read_tab(listing_book, _secret("LISTING_TAB", LISTING_TAB), LISTING_COLUMNS)

    trade_sheet = _secret("TRADE_SPREADSHEET_ID", listing_sheet)
    trade_tab = _secret("TRADE_TAB", TRADE_TAB)
    trade_book = client.open_by_key(_sheet_id(trade_sheet))
    try:
        trades = _read_tab(trade_book, trade_tab, TRADE_COLUMNS)
    except Exception:
        trades = pd.DataFrame()

    source = f"{_secret('LISTING_TAB', LISTING_TAB)} / {trade_tab}"
    return listings, trades, source
