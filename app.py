import os
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

import folium
from streamlit_folium import st_folium

import gspread
from google.oauth2.service_account import Credentials


# =========================
# 공개용 컬럼 Allowlist
# =========================
LISTING_ALLOW_COLUMNS = [
    "상태", "구역", "단지명", "동", "평형", "평형대", "대지지분", "층수", "가격",
    "요약내용", "위도", "경도",
    # 필요 시 노출(민감하면 제거): "부동산",
]

LOC_ALLOW_COLUMNS = ["단지명", "동", "위도", "경도", "구역"]
TRADE_ALLOW_COLUMNS = [
    "구역", "단지", "단지명", "단지명(단지)", "평형", "평형대",
    "날짜", "거래일", "계약일", "일자", "거래일자",
    "가격", "거래가격", "거래가", "실거래가", "금액", "거래금액",
    "동", "호", "비고"
]

# =========================
# Google Sheets 설정
# =========================
TAB_LISTING = "매매물건 목록"
TAB_LOC = "압구정 위치정보"
TAB_TRADE = "거래내역"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\n", "").strip() for c in df.columns]
    return df


def keep_allow_columns(df: pd.DataFrame, allow: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols = [c for c in allow if c in df.columns]
    return df[cols].copy()


def dong_key(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    m = re.findall(r"\d+", s)
    return m[0].lstrip("0") if m else s.strip()


def norm_area(x) -> str:
    """'1', '1구역', '01구역' => '1' 로 통일"""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    m = re.findall(r"\d+", s)
    if not m:
        return s
    return m[0].lstrip("0") or "0"


def norm_text(x: str) -> str:
    """단지명 비교용 정규화"""
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = s.replace("아파트", "").replace("apt", "").replace("apartment", "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[(){}\[\]\-_/·.,]", "", s)
    return s


def norm_size(x: str) -> str:
    """평형 비교용 정규화"""
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = s.replace("㎡", "").replace("m2", "").replace("m²", "").replace("평", "")
    s = re.sub(r"\s+", "", s)
    return s


def parse_pyeong_num(x) -> float | None:
    """'35평', '35', '35.5평' -> 35.5"""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def pyeong_bucket_10(pyeong: float | None) -> int | None:
    """35.5 -> 30 (30평대). NaN 안전 처리."""
    if pyeong is None or pd.isna(pyeong):
        return None
    return int(float(pyeong) // 10) * 10


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def fmt_decimal(x, nd=2) -> str:
    """59.500000 -> 59.5 / 62.100000 -> 62.1"""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    num = pd.to_numeric(x, errors="coerce")
    if pd.isna(num):
        return str(x)
    return f"{num:.{nd}f}".rstrip("0").rstrip(".")


def dataframe_height(df: pd.DataFrame, max_height: int = 700, row_height: int = 34, header_height: int = 42) -> int:
    n = 0 if df is None else int(len(df))
    h = header_height + (n * row_height)
    return max(160, min(h, max_height))


def st_df(obj, **kwargs):
    try:
        return st.dataframe(obj, hide_index=True, **kwargs)
    except TypeError:
        return st.dataframe(obj, **kwargs)


def compact_strings(df: "pd.DataFrame", max_len_by_col: dict | None = None, default_max: int = 10) -> "pd.DataFrame":
    """문자열 컬럼 축약(…): 모바일 폭 최적화"""
    if df is None or getattr(df, "empty", False):
        return df
    max_len_by_col = max_len_by_col or {}
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            max_len = int(max_len_by_col.get(col, default_max))
            ser = out[col].fillna("").astype(str)
            ser = ser.str.replace(r"\s+", " ", regex=True).str.strip()

            def _cut(x: str) -> str:
                if not x:
                    return ""
                if len(x) <= max_len:
                    return x
                if max_len <= 1:
                    return "…"
                return x[: max_len - 1] + "…"

            out[col] = ser.map(_cut)
    return out


def st_html_table(
    df: "pd.DataFrame",
    max_len_by_col: dict | None = None,
    default_max: int = 12,
    max_rows: int | None = None,
    col_widths: dict | None = None,
    wrapper_class: str = "tbl-wrap",
):
    """Streamlit 표를 HTML로 렌더링 (가운데 정렬 + 스크롤 컨테이너).
    - 문자열 컬럼은 compact_strings로 축약
    - col_widths로 열별 width 강제
    - wrapper_class로 컨테이너 높이/스크롤 정책 제어
    """
    if df is None or getattr(df, "empty", False):
        st.info("표로 표시할 데이터가 없습니다.")
        return

    disp = compact_strings(df.copy(), max_len_by_col=max_len_by_col, default_max=default_max)
    if max_rows is not None:
        disp = disp.head(int(max_rows))

    html = disp.to_html(index=False, escape=True)

    if col_widths:
        cols = list(disp.columns)
        col_tags = []
        for c in cols:
            w = col_widths.get(c, None)
            if w:
                col_tags.append(f'<col style="width:{w};">')
            else:
                col_tags.append("<col>")
        colgroup = "<colgroup>" + "".join(col_tags) + "</colgroup>"
        html = re.sub(r"(<table[^>]*>)", r"\1" + colgroup, html, count=1)

    st.markdown(f'<div class="{wrapper_class}">{html}</div>', unsafe_allow_html=True)


def to_eok_display(value) -> str:
    """원 단위면 억으로 환산, 이미 억이면 그대로(표시만)"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return ""
    if num >= 1e8:
        num = num / 1e8
    return fmt_decimal(num, nd=2)


# =========================
# 추가: 숫자/가격(억)/지분당 가격 계산 유틸
# =========================
def parse_numeric_any(x) -> float | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    s = s.replace(",", "")
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def price_to_eok_num(x) -> float | None:
    v = parse_numeric_any(x)
    if v is None:
        return None
    if v >= 1e8:
        return v / 1e8
    return v


def fmt_eok(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return fmt_decimal(x, nd=2)


def make_circle_label_html(label: str, bg_color: str) -> str:
    size = 30
    return f"""
    <div style="
        background:{bg_color};
        width:{size}px;height:{size}px;
        border-radius:50%;
        border:2px solid rgba(0,0,0,0.45);
        display:flex;align-items:center;justify-content:center;
        font-weight:700;font-size:14px;
        color:#ffffff;
        box-shadow:0 1px 4px rgba(0,0,0,0.35);
        ">
        {label}
    </div>
    """


def pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def extract_spreadsheet_id(url_or_id: str) -> str:
    if not url_or_id:
        return ""
    s = str(url_or_id).strip()
    if "docs.google.com" not in s and "/" not in s and len(s) >= 20:
        return s
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", s)
    if m:
        return m.group(1)
    s = s.split("#", 1)[0]
    return s


def get_spreadsheet_id() -> str:
    if "SPREADSHEET_ID" in st.secrets:
        return extract_spreadsheet_id(st.secrets["SPREADSHEET_ID"])

    if "connections" in st.secrets and "gsheets" in st.secrets["connections"]:
        gs = st.secrets["connections"]["gsheets"]
        if "spreadsheet" in gs:
            sid = extract_spreadsheet_id(gs["spreadsheet"])
            if sid:
                return sid

    env = os.getenv("SPREADSHEET_ID")
    if env:
        return extract_spreadsheet_id(env)

    raise RuntimeError("SPREADSHEET_ID가 설정되지 않았습니다. Secrets 또는 환경변수를 설정하세요.")


def _repair_private_key_multiline_json(raw: str) -> str:
    s = raw.strip()
    m = re.search(r'"private_key"\s*:\s*"', s)
    if not m:
        return s

    start = m.end()
    end_match = re.search(r'"\s*,\s*"\s*client_email"\s*:', s[start:])
    if not end_match:
        end_match = re.search(r'"\s*,\s*"[a-zA-Z0-9_]+"\\s*:', s[start:])
        if not end_match:
            return s

    end = start + end_match.start()
    pk = s[start:end]
    pk2 = pk.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    return s[:start] + pk2 + s[end:]


def parse_service_account_json(value: str) -> dict:
    raw = str(value).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repaired = _repair_private_key_multiline_json(raw)
        return json.loads(repaired)


def get_service_account_info():
    if "GCP_SERVICE_ACCOUNT_JSON" in st.secrets:
        v = st.secrets["GCP_SERVICE_ACCOUNT_JSON"]
        if isinstance(v, dict):
            return v
        return parse_service_account_json(v)

    if "connections" in st.secrets and "gsheets" in st.secrets["connections"]:
        gs = st.secrets["connections"]["gsheets"]
        keys = [
            "type", "project_id", "private_key_id", "private_key",
            "client_email", "client_id",
            "auth_uri", "token_uri",
            "auth_provider_x509_cert_url", "client_x509_cert_url",
        ]
        sa = {k: gs[k] for k in keys if k in gs}
        required = ["type", "project_id", "private_key", "client_email", "token_uri"]
        if all(k in sa and str(sa[k]).strip() for k in required):
            return sa
        raise RuntimeError(
            "Streamlit Secrets의 [connections.gsheets]에 서비스계정 필수 항목이 부족합니다. "
            "type/project_id/private_key/client_email/token_uri를 확인하세요."
        )

    env = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if env:
        return parse_service_account_json(env)

    raise RuntimeError(
        "서비스계정이 설정되지 않았습니다. "
        "Streamlit Secrets에 GCP_SERVICE_ACCOUNT_JSON 또는 [connections.gsheets]를 등록하세요."
    )


@st.cache_data(ttl=600)
def load_data():
    sa = get_service_account_info()
    spreadsheet_id = get_spreadsheet_id()

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(sa, scopes=scopes)
    gc = gspread.authorize(creds)

    sh = gc.open_by_key(spreadsheet_id)

    ws_list = sh.worksheet(TAB_LISTING)
    ws_loc = sh.worksheet(TAB_LOC)

    df_list = pd.DataFrame(ws_list.get_all_records())
    df_loc = pd.DataFrame(ws_loc.get_all_records())

    try:
        ws_trade = sh.worksheet(TAB_TRADE)
        df_trade = pd.DataFrame(ws_trade.get_all_records())
    except Exception:
        df_trade = pd.DataFrame()

    df_list = clean_columns(df_list)
    df_loc = clean_columns(df_loc)
    df_trade = clean_columns(df_trade)

    df_list = keep_allow_columns(df_list, LISTING_ALLOW_COLUMNS)
    df_loc = keep_allow_columns(df_loc, LOC_ALLOW_COLUMNS)
    df_trade = keep_allow_columns(df_trade, TRADE_ALLOW_COLUMNS)

    return df_list, df_loc, df_trade, sa.get("client_email", "")


def build_grouped(df_active: pd.DataFrame) -> pd.DataFrame:
    g = (
        df_active.groupby(["단지명", "동_key"], dropna=False)
        .agg(
            구역=("구역", "first"),
            위도=("위도", "first"),
            경도=("경도", "first"),
            활성건수=("동_key", "size"),
        )
        .reset_index()
    )
    return g


def summarize_area_by_size(df_active: pd.DataFrame, area_value: str) -> pd.DataFrame:
    if not area_value:
        return pd.DataFrame()

    target = norm_area(area_value)
    df_area = df_active.copy()
    df_area["_area_norm"] = df_area["구역"].astype(str).map(norm_area)
    df_area = df_area[df_area["_area_norm"] == target].copy()
    if df_area.empty:
        return pd.DataFrame()

    size_key = "평형대" if "평형대" in df_area.columns else ("평형" if "평형" in df_area.columns else None)
    if not size_key:
        return pd.DataFrame()

    if "가격_num" not in df_area.columns:
        df_area["가격_num"] = df_area["가격"].map(price_to_eok_num)

    s = (
        df_area.groupby(size_key, dropna=False)
        .agg(
            매물건수=("가격_num", "size"),
            최저가격=("가격_num", "min"),
            최고가격=("가격_num", "max"),
        )
        .reset_index()
        .rename(columns={size_key: "평형"})
    )

    s["평형_sort"] = s["평형"].astype(str)
    s = s.sort_values(by="평형_sort").drop(columns=["평형_sort"]).reset_index(drop=True)

    for c in ["최저가격", "최고가격"]:
        s[c] = s[c].round(2)

    s["가격대(최저~최고)"] = s["최저가격"].map(fmt_eok) + " ~ " + s["최고가격"].map(fmt_eok)
    return s


def recent_trades(df_trade: pd.DataFrame, complex_name: str, pyeong_value: str, df_ref_share: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    - 단지명 + 평형(또는 평형대)만 일치하는 거래내역 최신 5건
    - 출력 컬럼: 날짜, 단지, 평형, 가격(억), 지분당 가격, 구역, 동, 층
    - 지분당 가격은 df_ref_share에서 (단지명+평형) 대표 대지지분을 매핑해 계산
    """
    if df_trade is None or df_trade.empty:
        return pd.DataFrame()

    col_complex = pick_first_existing_column(df_trade, ["단지", "단지명", "단지명(단지)"])
    col_size = pick_first_existing_column(df_trade, ["평형", "평형대"])
    col_date = pick_first_existing_column(df_trade, ["날짜", "거래일", "계약일", "일자", "거래일자"])
    if not (col_complex and col_size and col_date):
        return pd.DataFrame()

    t = df_trade.copy()
    t["_complex_norm"] = t[col_complex].astype(str).map(norm_text)
    t["_size_norm"] = t[col_size].astype(str).map(norm_size)

    complex_norm = norm_text(complex_name)
    size_norm = norm_size(pyeong_value)

    t = t[(t["_complex_norm"] == complex_norm) & (t["_size_norm"] == size_norm)].copy()
    if t.empty:
        return pd.DataFrame()

    t["_dt"] = pd.to_datetime(t[col_date], errors="coerce", format="%y.%m.%d")
    t.loc[t["_dt"].isna(), "_dt"] = pd.to_datetime(t.loc[t["_dt"].isna(), col_date], errors="coerce")

    t = t.dropna(subset=["_dt"]).sort_values("_dt", ascending=False).head(5).copy()

    price_col = pick_first_existing_column(t, ["가격", "거래가격", "거래가", "실거래가", "금액", "거래금액"])
    if price_col:
        t["_price_eok"] = t[price_col].map(price_to_eok_num)
        t["가격(억)"] = t["_price_eok"].map(fmt_eok)
    else:
        t["_price_eok"] = None
        t["가격(억)"] = ""

    if "호" in t.columns:
        t["층"] = t["호"].map(extract_floor_from_ho)
    elif "층" in t.columns:
        t["층"] = t["층"].map(extract_floor_from_level).map(lambda v: f"{v}층" if str(v).strip() else "")
    else:
        t["층"] = ""

    # 지분당 가격(매물 목록에서 지분 매핑)
    t["_share_num"] = None
    if df_ref_share is not None and not df_ref_share.empty:
        ref = df_ref_share.copy()
        if "단지명" in ref.columns and "평형" in ref.columns and "대지지분_num" in ref.columns:
            ref["_complex_norm"] = ref["단지명"].astype(str).map(norm_text)
            ref["_size_norm"] = ref["평형"].astype(str).map(norm_size)

            share_map = (
                ref.dropna(subset=["대지지분_num"])
                .groupby(["_complex_norm", "_size_norm"], dropna=False)["대지지분_num"]
                .median()
                .to_dict()
            )
            t["_share_num"] = t.apply(lambda r: share_map.get((r["_complex_norm"], r["_size_norm"])), axis=1)

    t["_ppshare"] = t.apply(
        lambda r: (r["_price_eok"] / r["_share_num"])
        if (pd.notna(r.get("_price_eok")) and pd.notna(r.get("_share_num")) and float(r["_share_num"]) != 0)
        else None,
        axis=1,
    )
    t["지분당 가격"] = t["_ppshare"].map(fmt_eok)

    out = pd.DataFrame()
    out["날짜"] = t[col_date].astype(str)
    out["단지"] = t[col_complex].astype(str)
    out["평형"] = t[col_size].astype(str)
    out["가격(억)"] = t["가격(억)"].astype(str)
    out["지분당 가격"] = t["지분당 가격"].astype(str)

    out["구역"] = t["구역"].astype(str) if "구역" in t.columns else ""
    out["동"] = t["동"].astype(str) if "동" in t.columns else ""
    out["층"] = t["층"].astype(str)

    return out[["날짜", "단지", "평형", "가격(억)", "지분당 가격", "구역", "동", "층"]]


def resolve_clicked_meta(clicked_lat, clicked_lng, marker_rows):
    if clicked_lat is None or clicked_lng is None:
        return None
    clat = float(clicked_lat)
    clng = float(clicked_lng)

    best_meta = None
    best_d = None
    for lat, lng, meta in marker_rows:
        d = (float(lat) - clat) ** 2 + (float(lng) - clng) ** 2
        if best_d is None or d < best_d:
            best_d = d
            best_meta = meta
    return best_meta


def extract_floor_from_level(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else s


def extract_floor_from_ho(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""

    if "층" in s:
        m = re.search(r"(\d+)\s*층", s)
        if m:
            return f"{int(m.group(1))}층"
        m2 = re.search(r"(\d+)", s)
        return f"{int(m2.group(1))}층" if m2 else s

    if "/" in s or "-" in s:
        m = re.search(r"(\d+)", s)
        if m:
            return f"{int(m.group(1))}층"
        return ""

    digits = re.sub(r"\D", "", s)
    if not digits:
        return ""
    if len(digits) >= 3:
        floor_digits = digits[:-2]
        try:
            return f"{int(floor_digits)}층"
        except Exception:
            return ""
    try:
        return f"{int(digits)}층"
    except Exception:
        return ""


# =================== UI ===================
st.set_page_config(layout="wide")

# ===== 상단: 타이틀(좌) + 업소홍보(우) =====
col_title, col_promo = st.columns([3.3, 1.7])

with col_title:
    st.markdown('<div class="app-title"><h1>압구정동 시세 지도</h1></div>', unsafe_allow_html=True)

with col_promo:
    st.markdown(
        """
        <div class="promo-card">
            <div class="promo-name">☎ 압구정 원 부동산</div>
            <div class="promo-desc">압구정 재건축 전문 컨설팅 · 확실한 순위 판단</div>
            <div class="promo-label">문의</div>
            <div class="promo-contact">02-540-3334 / 최이사 Mobile 010-3065-1780</div>
            <div class="promo-sub">압구정 미래가치 예측.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
<style>

/* ===== Header title & Promo card ===== */
.app-title h1{
    margin: 0;
    padding: 0;
    font-size: 56px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #2b2f36;
}
.promo-card{
    border: 1px solid rgba(0,0,0,0.10);
    border-radius: 14px;
    padding: 14px 16px;
    background: #ffffff;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}
.promo-name{
    font-size: 18px;
    font-weight: 800;
    margin: 0 0 6px 0;
}
.promo-desc{
    font-size: 14px;
    font-weight: 600;
    color: #2b2f36;
    margin: 0 0 10px 0;
}
.promo-label{
    font-size: 13px;
    font-weight: 800;
    margin: 0 0 2px 0;
}
.promo-contact{
    font-size: 13px;
    font-weight: 600;
    color: #2b2f36;
    margin: 0;
    line-height: 1.4;
}
.promo-sub{
    font-size: 12px;
    color: rgba(0,0,0,0.55);
    margin: 8px 0 0 0;
}

/* ===== 지도 높이 ===== */
div[data-testid="stIFrame"],
div[data-testid="stIFrame"] iframe {
    height: 455px !important;
    width: 100% !important;
}
@media (max-width: 768px) {
    div[data-testid="stIFrame"],
    div[data-testid="stIFrame"] iframe {
        height: 300px !important;
    }
}

/* ===== 기본 표 래퍼 ===== */
.tbl-wrap{
    width: 100%;
    max-height: 520px;
    overflow-y: auto;
    overflow-x: hidden;
    -webkit-overflow-scrolling: touch;
}
@media (max-width: 768px){
    .tbl-wrap{
        max-height: 360px;
    }
}

/* ===== Top(전체 지분당 가격) : 약 15행만 보이는 스크롤 박스 ===== */
.tbl-wrap-15{
    width: 100%;
    max-height: 460px;      /* PC 기준: 약 15행 */
    overflow-y: auto;
    overflow-x: hidden;
    -webkit-overflow-scrolling: touch;
}
@media (max-width: 768px){
    .tbl-wrap-15{
        max-height: 320px;  /* 모바일 기준: 약 15행 */
    }
}

/* ===== 표 공통 스타일 (tbl-wrap / tbl-wrap-15 모두 적용) ===== */
.tbl-wrap table,
.tbl-wrap-15 table{
    width: 100% !important;
    table-layout: fixed !important;
    border-collapse: collapse;
}
.tbl-wrap th, .tbl-wrap td,
.tbl-wrap-15 th, .tbl-wrap-15 td{
    text-align: center !important;
    vertical-align: middle !important;
    border: 1px solid rgba(0,0,0,0.10);
    padding: 5px 3px;
    font-size: 12px;
    line-height: 1.25;
    max-width: 0;
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: break-word;
}
.tbl-wrap th,
.tbl-wrap-15 th{
    background: rgba(0,0,0,0.03);
    font-weight: 800;
}
@media (max-width: 768px){
    .tbl-wrap th, .tbl-wrap td,
    .tbl-wrap-15 th, .tbl-wrap-15 td{
        padding: 3px 2px;
        font-size: 10.5px;
    }
}

</style>
    """,
    unsafe_allow_html=True,
)

only_active = True

if st.button("데이터 새로고침"):
    load_data.clear()
    st.rerun()

# ====== Load ======
df_list, df_loc, df_trade, _client_email = load_data()

# --- 요약내용 컬럼명 표준화/보장 ---
df_list = df_list.copy()
rename_map = {}
for c in df_list.columns:
    c0 = str(c)
    c_norm = re.sub(r"\s+", "", c0)
    if c_norm == "요약내용" or ("요약" in c_norm and "내용" in c_norm):
        rename_map[c] = "요약내용"

if rename_map:
    df_list.rename(columns=rename_map, inplace=True)

summary_src = pick_first_existing_column(df_list, ["요약내용", "요약 내용", "요약", "설명", "비고", "메모"])
if "요약내용" not in df_list.columns:
    df_list["요약내용"] = df_list[summary_src] if summary_src else ""
elif summary_src and summary_src != "요약내용":
    left = df_list["요약내용"].astype(str).str.strip()
    df_list.loc[left.eq(""), "요약내용"] = df_list.loc[left.eq(""), summary_src]
# -------------------------------

need_cols = ["평형대", "구역", "단지명", "평형", "대지지분", "동", "층수", "가격", "상태"]
missing = [c for c in need_cols if c not in df_list.columns]
if missing:
    st.error(f"'매매물건 목록' 탭에서 다음 컬럼이 필요합니다: {missing}")
    st.stop()

for c in ["위도", "경도"]:
    if c not in df_list.columns:
        df_list[c] = None

df_list["동_key"] = df_list["동"].apply(dong_key)
df_list["층"] = df_list["층수"].apply(extract_floor_from_level)

df_loc = df_loc.copy()
if "동" in df_loc.columns:
    df_loc["동_key"] = df_loc["동"].apply(dong_key)

df_view = df_list.copy()
if only_active:
    df_view = df_view[df_view["상태"].astype(str).str.strip() == "활성"].copy()

if df_view.empty:
    st.warning("현재 표시할 매물이 없습니다.")
    st.stop()

df_view["위도"] = df_view["위도"].apply(to_float)
df_view["경도"] = df_view["경도"].apply(to_float)

if all(c in df_loc.columns for c in ["단지명", "동_key", "위도", "경도"]):
    df_loc = df_loc.copy()
    df_loc["위도"] = df_loc["위도"].apply(to_float)
    df_loc["경도"] = df_loc["경도"].apply(to_float)

    df_view = df_view.merge(
        df_loc[["단지명", "동_key", "위도", "경도"]].rename(columns={"위도": "위도_loc", "경도": "경도_loc"}),
        on=["단지명", "동_key"],
        how="left",
    )
    df_view["위도"] = df_view["위도"].fillna(df_view["위도_loc"])
    df_view["경도"] = df_view["경도"].fillna(df_view["경도_loc"])
    df_view.drop(columns=["위도_loc", "경도_loc"], inplace=True)

# =========================
# 가격/평형대/대지지분/지분당 가격 정규화 컬럼
# =========================
df_view["가격_eok_num"] = df_view["가격"].map(price_to_eok_num)
df_view["가격_num"] = df_view["가격_eok_num"]

df_view["평형대_num"] = df_view["평형대"].map(parse_pyeong_num)
df_view["평형대_bucket"] = df_view["평형대_num"].apply(pyeong_bucket_10)

df_view["대지지분_num"] = df_view["대지지분"].map(parse_numeric_any)

df_view["지분당가격_num"] = df_view.apply(
    lambda r: (r["가격_eok_num"] / r["대지지분_num"])
    if (pd.notna(r.get("가격_eok_num")) and pd.notna(r.get("대지지분_num")) and float(r["대지지분_num"]) != 0)
    else None,
    axis=1,
)

df_view["가격(억)"] = df_view["가격_eok_num"].map(fmt_eok)
df_view["지분당 가격"] = df_view["지분당가격_num"].map(fmt_eok)

# 지도/마커용 데이터프레임(좌표가 있는 매물만)
df_map = df_view.dropna(subset=["위도", "경도"]).copy()

# 그룹(동 단위 포인트)
gdf = (
    df_map.groupby(["단지명", "동_key"], dropna=False)
    .agg(
        구역=("구역", "first"),
        위도=("위도", "first"),
        경도=("경도", "first"),
        활성건수=("동_key", "size"),
    )
    .reset_index()
)

palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
areas = sorted([a for a in gdf["구역"].dropna().astype(str).unique()])
area_color = {a: palette[i % len(palette)] for i, a in enumerate(areas)}
default_color = "#333333"

DEFAULT_ZOOM = 16

if "map_center" not in st.session_state:
    try:
        lat0 = float(pd.to_numeric(gdf["위도"], errors="coerce").mean())
        lng0 = float(pd.to_numeric(gdf["경도"], errors="coerce").mean())
        if pd.isna(lat0) or pd.isna(lng0):
            raise ValueError("NaN center")
        st.session_state["map_center"] = [lat0, lng0]
    except Exception:
        st.session_state["map_center"] = [37.5275, 127.0300]
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = DEFAULT_ZOOM
if "selected_meta" not in st.session_state:
    st.session_state["selected_meta"] = None
if "last_click_sig" not in st.session_state:
    st.session_state["last_click_sig"] = ""
if "quick_filter_mode" not in st.session_state:
    st.session_state["quick_filter_mode"] = "size"
if "quick_filter_bucket" not in st.session_state:
    st.session_state["quick_filter_bucket"] = 30
if "quick_filter_area_norm" not in st.session_state:
    st.session_state["quick_filter_area_norm"] = "__ALL__"

# ====== 지도 생성 ======
m = folium.Map(
    location=st.session_state["map_center"],
    zoom_start=int(st.session_state["map_zoom"]),
    tiles="CartoDB positron",
)

marker_rows = []
for _, r in gdf.iterrows():
    marker_rows.append(
        (
            r["위도"],
            r["경도"],
            {"단지명": r["단지명"], "동_key": r["동_key"], "구역": r["구역"], "위도": r["위도"], "경도": r["경도"]},
        )
    )

for _, r in gdf.iterrows():
    area_raw = str(r["구역"]) if pd.notna(r["구역"]) else ""
    bg = area_color.get(area_raw, default_color)
    dong_label = str(r["동_key"])
    area_display = f"{norm_area(area_raw)}구역" if norm_area(area_raw) else ""
    tooltip = f"{area_display} | {r['단지명']} {dong_label}동 | 활성 {int(r['활성건수'])}건"

    folium.CircleMarker(
        location=[r["위도"], r["경도"]],
        radius=18,
        weight=0,
        opacity=0,
        fill=True,
        fill_opacity=0,
        tooltip=tooltip,
    ).add_to(m)

    folium.Marker(
        location=[r["위도"], r["경도"]],
        icon=folium.DivIcon(html=make_circle_label_html(dong_label, bg)),
        tooltip=tooltip,
    ).add_to(m)

st.subheader("지도")
out = st_folium(
    m,
    height=455,
    width=None,
    returned_objects=["last_object_clicked"],
    key="map",
)

if out:
    clicked = out.get("last_object_clicked", None)
    if clicked:
        lat = clicked.get("lat")
        lng = clicked.get("lng")
        if lat is not None and lng is not None:
            click_sig = f"{round(float(lat), 6)},{round(float(lng), 6)}"
            if st.session_state["last_click_sig"] != click_sig:
                meta = resolve_clicked_meta(lat, lng, marker_rows)
                if meta:
                    st.session_state["selected_meta"] = meta
                    st.session_state["map_center"] = [float(meta["위도"]), float(meta["경도"])]
                    st.session_state["map_zoom"] = int(st.session_state.get("map_zoom") or DEFAULT_ZOOM)
                    st.session_state["last_click_sig"] = click_sig
                    st.rerun()

meta = st.session_state.get("selected_meta", None)
if not meta:
    st.info("지도에서 마커를 클릭하면 아래에 상세가 표시됩니다.")
    st.stop()

complex_name = meta["단지명"]
area_value = str(meta["구역"]) if pd.notna(meta["구역"]) else ""

# =========================
# (0) 전체지역: 지분당 가격 (싼 순) - 15행 노출 박스 + 스크롤로 전체
# =========================
st.subheader("지분당 가성비 - 전체지역/전체평형")

df_top = df_view.copy()
df_top = df_top[df_top["지분당가격_num"].notna()].copy()
df_top = df_top[df_top["지분당가격_num"] > 0].copy()

if df_top.empty:
    st.info("지분당 가격을 계산할 수 있는 매물이 없습니다. (가격/대지지분 값 확인 필요)")
else:
    df_top = df_top.sort_values(["지분당가격_num", "가격_num"], ascending=[True, True]).copy()

    top_cols = ["구역", "단지명", "평형", "대지지분", "동", "층", "가격(억)", "지분당 가격"]
    top_cols = [c for c in top_cols if c in df_top.columns]

    st_html_table(
        df_top[top_cols].reset_index(drop=True),
        default_max=10,
        max_len_by_col={"단지명": 10, "동": 4, "평형": 6, "대지지분": 8},
        col_widths={
            "구역": "8%",
            "단지명": "20%",
            "평형": "10%",
            "대지지분": "12%",
            "동": "8%",
            "층": "8%",
            "가격(억)": "14%",
            "지분당 가격": "20%",
        },
        wrapper_class="tbl-wrap-15",
    )

st.divider()

# =========================
# (1) 단지 + 평형 선택
# =========================
df_complex = df_view[df_view["단지명"] == complex_name].copy()
pyeong_candidates = sorted(df_complex["평형"].astype(str).str.strip().dropna().unique().tolist())

st.subheader("선택 단지/평형의 매물시세")
if not pyeong_candidates:
    st.info("선택한 단지에서 평형 정보를 찾을 수 없습니다.")
    st.stop()

sel_key = f"sel_pyeong_{norm_text(complex_name)}"
sel_pyeong = st.selectbox("평형 선택", pyeong_candidates, index=0, key=sel_key)

size_norm = norm_size(sel_pyeong)
df_pick = df_complex.copy()
df_pick["_size_norm"] = df_pick["평형"].astype(str).map(norm_size)
df_pick = df_pick[df_pick["_size_norm"] == size_norm].copy()

df_pick = df_pick.sort_values(
    ["지분당가격_num", "가격_num"],
    ascending=[True, True],
    na_position="last"
).reset_index(drop=True)

show_cols = ["단지명", "평형", "대지지분", "동", "층", "가격(억)", "지분당 가격"]
if "부동산" in df_pick.columns:
    show_cols.insert(show_cols.index("가격(억)") + 1, "부동산")
show_cols = [c for c in show_cols if c in df_pick.columns]
view_pick = df_pick[show_cols].reset_index(drop=True)

st_html_table(
    view_pick,
    max_len_by_col={"단지명": 10, "동": 4, "평형": 6, "대지지분": 8, "부동산": 8},
    default_max=10,
    col_widths={
        "단지명": "20%",
        "평형": "10%",
        "대지지분": "12%",
        "동": "8%",
        "층": "8%",
        "가격(억)": "18%",
        "지분당 가격": "24%",
        "부동산": "20%",
    },
)

st.divider()

# =========================
# (2) 거래내역 최근 5건 (+지분당 가격)
# =========================
st.subheader("거래내역 최근 5건")
trades = recent_trades(df_trade, complex_name, sel_pyeong, df_ref_share=df_view)
if trades.empty:
    st.info("일치하는 거래내역이 없습니다.")
else:
    trades2 = trades.reset_index(drop=True)
    st_html_table(
        trades2,
        default_max=12,
        col_widths={
            "날짜": "14%",
            "단지": "22%",
            "평형": "10%",
            "가격(억)": "12%",
            "지분당 가격": "14%",
            "구역": "8%",
            "동": "8%",
            "층": "12%",
        },
    )

st.divider()

# =========================
# (3) 좌: 선택 구역 요약
# (4) 우: 빠른 필터
# =========================
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("선택 구역 평형별 시세")
    if not area_value:
        st.info("선택한 마커의 구역 정보가 없습니다.")
    else:
        summary = summarize_area_by_size(df_view, area_value)
        if summary.empty:
            st.info("해당 구역에서 요약할 데이터가 없습니다.")
        else:
            summary_view = summary[["평형", "매물건수", "가격대(최저~최고)", "최저가격", "최고가격"]].reset_index(drop=True)
            st_html_table(summary_view, default_max=12)

with col_right:
    st.subheader("압구정 매물 시세 검색")

    row1 = st.columns(4)
    row2 = st.columns(4)

    buckets_row1 = [20, 30, 40, 50]
    buckets_row2 = [60, 70, 80]

    for col, b in zip(row1, buckets_row1):
        if col.button(f"{b}평대", use_container_width=True, key=f"qsize_{b}"):
            st.session_state["quick_filter_mode"] = "size"
            st.session_state["quick_filter_bucket"] = b
            st.rerun()

    for col, b in zip(row2[:3], buckets_row2):
        if col.button(f"{b}평대", use_container_width=True, key=f"qsize_{b}"):
            st.session_state["quick_filter_mode"] = "size"
            st.session_state["quick_filter_bucket"] = b
            st.rerun()

    if row2[3].button("가격순", use_container_width=True, key="qprice"):
        st.session_state["quick_filter_mode"] = "price"
        st.rerun()

    mode = st.session_state["quick_filter_mode"]

    df_area = df_view[["구역"]].copy()
    df_area["_area_norm"] = df_area["구역"].astype(str).map(norm_area)
    df_area = df_area[df_area["_area_norm"].astype(str).str.strip() != ""].copy()

    area_norms = sorted(
        df_area["_area_norm"].dropna().astype(str).unique().tolist(),
        key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)),
    )

    area_labels = {an: (f"{int(an)}구역" if str(an).isdigit() else str(an)) for an in area_norms}

    sel_area_norm = st.session_state.get("quick_filter_area_norm", "__ALL__")
    if sel_area_norm != "__ALL__" and sel_area_norm not in area_labels:
        sel_area_norm = "__ALL__"
        st.session_state["quick_filter_area_norm"] = "__ALL__"

    st.markdown("**구역 선택**")

    area_btn_labels = ["__ALL__"] + area_norms
    area_btn_text = ["전체"] + [area_labels.get(an, f"{an}구역") for an in area_norms]

    max_cols = 4
    for start in range(0, len(area_btn_labels), max_cols):
        cols = st.columns(max_cols)
        chunk_labels = area_btn_labels[start:start + max_cols]
        chunk_text = area_btn_text[start:start + max_cols]

        for i, (lab, txt) in enumerate(zip(chunk_labels, chunk_text)):
            is_sel = (sel_area_norm == "__ALL__") if lab == "__ALL__" else (lab == sel_area_norm)
            shown = f"✓ {txt}" if is_sel else txt
            key = "qarea_all" if lab == "__ALL__" else f"qarea_{lab}"
            if cols[i].button(shown, use_container_width=True, key=key):
                st.session_state["quick_filter_area_norm"] = "__ALL__" if lab == "__ALL__" else lab
                st.rerun()

    area_display2 = "전체" if sel_area_norm == "__ALL__" else area_labels.get(sel_area_norm, f"{sel_area_norm}구역")

    if mode == "size":
        st.caption(f"현재: {area_display2} / {st.session_state['quick_filter_bucket']}평대 (가격 낮은 순)")
    else:
        st.caption(f"현재: {area_display2} / 전체 평형 (가격 낮은 순)")

    dfq = df_view.copy()
    dfq = dfq[dfq["가격_num"].notna()].copy()
    dfq["_area_norm"] = dfq["구역"].astype(str).map(norm_area)

    if sel_area_norm != "__ALL__":
        dfq = dfq[dfq["_area_norm"] == sel_area_norm].copy()

    if mode == "size":
        b = st.session_state["quick_filter_bucket"]
        dfq = dfq[dfq["평형대_bucket"] == b].copy()

    dfq = dfq.sort_values("가격_num", ascending=True).reset_index(drop=True)

    if dfq.empty:
        st.info("조건에 맞는 매물이 없습니다.")
    else:
        display_cols = ["구역", "평형대", "평형", "단지명", "대지지분", "동", "층", "가격(억)", "지분당 가격"]
        cols_exist = [c for c in display_cols if c in dfq.columns]

        st.caption(f"해당 조건 매물: {len(dfq):,}건 (표는 스크롤로 전체 확인 가능합니다)")
        st.markdown("낮은 가격 순서")

        df_show = dfq[cols_exist + ["위도", "경도", "동_key", "가격_num", "지분당가격_num"]].copy().reset_index(drop=True)
        df_table = df_show[cols_exist].copy()

        st_html_table(
            df_table,
            max_len_by_col={"단지명": 10, "동": 4, "평형": 6, "평형대": 6, "대지지분": 8},
            default_max=10,
            col_widths={
                "구역": "8%",
                "평형대": "10%",
                "평형": "10%",
                "단지명": "18%",
                "대지지분": "12%",
                "동": "8%",
                "층": "8%",
                "가격(억)": "13%",
                "지분당 가격": "13%",
            },
        )
