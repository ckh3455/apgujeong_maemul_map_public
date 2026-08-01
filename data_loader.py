import json
import os
import re

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials


LISTING_TAB = "매매물건 목록"
TRADE_TAB = "거래내역"
UNIT_MASTER_SPREADSHEET_ID = "1QGSM-mICX9KYa5Izym6sFKVaWwO-o0j86V-KmJ-w0IM"
UNIT_MASTER_TAB = "공동주택 공시가격"

LISTING_COLUMNS = ["상태", "구역", "단지명", "동", "평형", "평형대", "대지지분", "층수", "가격"]
UNIT_COLUMNS = ["구역", "주소", "단지명", "전용면적(㎡)", "대지지분(평)", "특기사항", "평형", "동", "호"]
TRADE_COLUMNS = [
    "구역", "단지", "단지명", "단지명(단지)", "평형", "평형대", "전용면적",
    "날짜", "거래일", "계약일", "일자", "거래일자", "계약년월", "계약일자",
    "계약년", "계약월",
    "가격", "거래가격", "거래가", "실거래가", "금액", "거래금액", "거래금액(만원)",
    "전용면적(㎡)", "동", "호", "층", "해제여부", "해제사유발생일", "거래유형", "비고",
]


def _sheet_id(value: str) -> str:
    match = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", str(value))
    return match.group(1) if match else str(value).strip()


def _repair_private_key_json(raw: str) -> str:
    """JSON 문자열 안 private_key의 실제 줄바꿈만 JSON용 \\n으로 복구한다."""
    text = raw.strip()
    marker = re.search(r'"private_key"\s*:\s*"', text)
    if not marker:
        return text
    start = marker.end()
    end_match = re.search(r'"\s*,\s*"(?:client_email|client_id|auth_uri)"\s*:', text[start:])
    if not end_match:
        return text
    end = start + end_match.start()
    private_key = text[start:end]
    private_key = private_key.replace("\r\n", "\n").replace("\r", "\n")
    private_key = private_key.replace("\n", "\\n")
    return text[:start] + private_key + text[end:]


def _parse_service_account(value) -> dict:
    if isinstance(value, dict):
        return dict(value)
    raw = str(value).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_repair_private_key_json(raw))


def _service_account() -> dict:
    if "GCP_SERVICE_ACCOUNT_JSON" in st.secrets:
        return _parse_service_account(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])
    env = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if env:
        return _parse_service_account(env)
    raise RuntimeError("GCP_SERVICE_ACCOUNT_JSON이 없습니다.")


def _secret(name: str, default=None):
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)


def _frame_from_values(values) -> pd.DataFrame:
    """빈/중복 헤더가 있는 시트도 고유한 임시 열 이름으로 안전하게 읽는다."""
    if not values:
        return pd.DataFrame()
    raw_headers = [str(c).replace("\n", "").strip() for c in values[0]]
    seen = {}
    headers = []
    for index, header in enumerate(raw_headers):
        base = header or f"__blank_{index + 1}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        headers.append(base if count == 0 else f"{base}__{count + 1}")
    width = len(headers)
    rows = [(list(row) + [""] * width)[:width] for row in values[1:]]
    return pd.DataFrame(rows, columns=headers)


def _read_tab(book, tab_name: str, allow: list[str]) -> pd.DataFrame:
    frame = _frame_from_values(book.worksheet(tab_name).get_all_values())
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
    unit_sheet = _secret("UNIT_MASTER_SPREADSHEET_ID", UNIT_MASTER_SPREADSHEET_ID)
    unit_tab = _secret("UNIT_MASTER_TAB", UNIT_MASTER_TAB)
    unit_book = client.open_by_key(_sheet_id(unit_sheet))
    units = _read_tab(unit_book, unit_tab, UNIT_COLUMNS)

    # 가격 판단은 매물 원본 파일 안에서 중개사가 정리한 `거래내역`만 사용한다.
    # 외부 `압구정동 거래데이터`와 TRADE_SPREADSHEET_ID 설정은 가격 계산에 사용하지 않는다.
    trade_tab = TRADE_TAB
    trades = _read_tab(listing_book, trade_tab, TRADE_COLUMNS)

    source = f"{unit_tab} / {_secret('LISTING_TAB', LISTING_TAB)} / {trade_tab}"
    return listings, units, trades, source
