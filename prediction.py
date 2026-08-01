import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


def first_column(df, candidates):
    return next((c for c in candidates if c in df.columns), None)


def normalize_area(value):
    match = re.search(r"\d+", str(value or ""))
    return (match.group(0).lstrip("0") or "0") if match else ""


def normalize_text(value):
    text = str(value or "").lower().replace("아파트", "").replace("apt", "")
    return re.sub(r"[\s(){}\[\]\-_/·.,]", "", text)


def complex_family(value):
    text = normalize_text(value)
    if "신현대" in text:
        return "신현대"
    if "미성" in text:
        match = re.search(r"미성(1|2)", text)
        return f"미성{match.group(1)}" if match else "미성"
    if "한양" in text:
        return "한양"
    if "대림" in text:
        return "대림"
    if "현대" in text:
        match = re.search(r"현대(\d+)", text)
        return f"현대{match.group(1)}" if match else "현대"
    return re.sub(r"\d+차.*$", "", text)


def normalize_size(value):
    text = str(value or "").lower().replace("평", "").replace("㎡", "").replace("m²", "").replace("m2", "")
    match = re.search(r"\d+(?:\.\d+)?", text)
    return f"{float(match.group()):g}" if match else text.strip()


def dong_number(value):
    match = re.search(r"\d+", str(value or ""))
    return (match.group().lstrip("0") or "0") if match else ""


def parse_number(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None


def price_eok(value):
    number = parse_number(value)
    if number is None:
        return None
    if number >= 100000000:
        return number / 100000000
    if number >= 10000:  # 만원 단위 실거래
        return number / 10000
    return number


def floor_from_ho(value):
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) < 3:
        return None
    floor = int(digits[:-2])
    return floor if floor > 0 else None


def floor_from_value(value):
    number = parse_number(value)
    return int(number) if number is not None and number > 0 else None


def parse_listing_floor(value):
    text = str(value or "")
    numbers = [int(x) for x in re.findall(r"\d+", text)]
    if not numbers:
        return None, None
    if len(numbers) >= 2:
        return numbers[0], numbers[-1]
    return numbers[0], None


def parse_date(value):
    if value is None or str(value).strip() == "":
        return pd.NaT
    text = str(value).strip()
    digits = re.sub(r"\D", "", text)
    if len(digits) == 8:
        return pd.to_datetime(digits, format="%Y%m%d", errors="coerce")
    match = re.fullmatch(r"(\d{2})[./-](\d{1,2})[./-](\d{1,2})", text)
    if match:
        return pd.Timestamp(2000 + int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return pd.to_datetime(text, errors="coerce")


def floor_band(floor, total_floor=None):
    if not floor:
        return "unknown"
    if floor == 1:
        return "first"
    if total_floor and total_floor > 0:
        ratio = floor / total_floor
        if ratio <= 0.25:
            return "low"
        if ratio <= 0.65:
            return "middle"
        if ratio < 0.92:
            return "high"
        return "top"
    if floor <= 3:
        return "low"
    if floor <= 8:
        return "middle"
    return "high"


def prepare_listings(df):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for required in ["구역", "단지명", "동", "평형", "층수", "가격"]:
        if required not in out.columns:
            out[required] = ""
    if "상태" in out.columns:
        active = out["상태"].astype(str).str.strip()
        out = out[(active == "") | (active == "활성")].copy()
    out["_area"] = out["구역"].map(normalize_area)
    out["_complex"] = out["단지명"].map(normalize_text)
    out["_family"] = out["단지명"].map(complex_family)
    out["_dong"] = out["동"].map(dong_number)
    out["_size"] = out["평형"].map(normalize_size)
    out["_price"] = out["가격"].map(price_eok)
    parsed = out["층수"].map(parse_listing_floor)
    out["_floor"] = parsed.map(lambda x: x[0])
    out["_total_floor"] = parsed.map(lambda x: x[1])
    return out


def prepare_units(df):
    """공동주택 공시가격 탭을 구역-동-호수 기준 마스터로 정규화한다."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for required in ["구역", "단지명", "전용면적(㎡)", "평형", "동", "호"]:
        if required not in out.columns:
            out[required] = ""
    out["_area"] = out["구역"].map(normalize_area)
    out["_complex"] = out["단지명"].map(normalize_text)
    out["_family"] = out["단지명"].map(complex_family)
    out["_dong"] = out["동"].map(dong_number)
    out["_ho"] = out["호"].map(lambda x: re.sub(r"\D", "", str(x)) if pd.notna(x) else "")
    out["_floor"] = out["_ho"].map(floor_from_ho)
    out["_size"] = out["평형"].map(normalize_size)
    out["_sqm"] = out["전용면적(㎡)"].map(parse_number)
    out = out[(out["_area"] != "") & (out["_dong"] != "") & (out["_ho"] != "")].copy()
    out["_total_floor"] = out.groupby(["_area", "_dong"])["_floor"].transform("max")
    return out


def prepare_trades(df, listings, units=None):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    complex_col = first_column(out, ["단지", "단지명", "단지명(단지)"])
    size_col = first_column(out, ["평형", "평형대", "전용면적", "전용면적(㎡)"])
    date_col = first_column(out, ["거래일", "날짜", "일자", "거래일자", "계약일자"])
    price_col = first_column(out, ["거래가격", "거래가", "실거래가", "거래금액", "거래금액(만원)", "금액", "가격"])
    split_date = all(c in out.columns for c in ["계약년", "계약월", "계약일"])
    if not all([complex_col, size_col, price_col]) or (not date_col and not split_date):
        return pd.DataFrame()
    out["_complex"] = out[complex_col].map(normalize_text)
    out["_family"] = out[complex_col].map(complex_family)
    out["_size"] = out[size_col].map(normalize_size)
    out["_sqm"] = out[size_col].map(parse_number)
    # 국토부 전용면적을 공시가격 마스터의 공식 평형명으로 변환한다.
    # 예: 신현대 + 108.88㎡ -> 신현대 35평
    if units is not None and not units.empty:
        unit_map = units.dropna(subset=["_sqm"]).copy()
        unit_map["_sqm_key"] = unit_map["_sqm"].round(2)
        size_map = (
            unit_map.groupby(["_family", "_sqm_key"])["_size"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
            .to_dict()
        )
        out["_size"] = [
            size_map.get((family, round(sqm, 2)), old_size) if pd.notna(sqm) else old_size
            for family, sqm, old_size in zip(out["_family"], out["_sqm"], out["_size"])
        ]
    if split_date:
        out["_date"] = pd.to_datetime(
            dict(
                year=pd.to_numeric(out["계약년"], errors="coerce"),
                month=pd.to_numeric(out["계약월"], errors="coerce"),
                day=pd.to_numeric(out["계약일"], errors="coerce"),
            ),
            errors="coerce",
        )
    else:
        out["_date"] = out[date_col].map(parse_date)
    out["_price"] = out[price_col].map(price_eok)
    out["_dong"] = out["동"].map(dong_number) if "동" in out else ""
    if "층" in out:
        out["_floor"] = out["층"].map(floor_from_value)
    elif "호" in out:
        out["_floor"] = out["호"].map(floor_from_ho)
    else:
        out["_floor"] = None
    cancel_col = first_column(out, ["해제여부", "해제사유발생일"])
    if cancel_col:
        cancelled = ~out[cancel_col].astype(str).str.strip().isin(["", "-", "nan", "None"])
        out = out[~cancelled].copy()
    out = out[out["_date"].notna() & out["_price"].notna() & (out["_price"] > 0)].copy()

    totals = listings.dropna(subset=["_total_floor"]).groupby(["_complex", "_dong"])["_total_floor"].median()
    out["_total_floor"] = [totals.get((c, d), np.nan) for c, d in zip(out["_complex"], out["_dong"])]
    out["_band"] = [floor_band(f, t if pd.notna(t) else None) for f, t in zip(out["_floor"], out["_total_floor"])]
    out["_month"] = out["_date"].dt.to_period("M")
    return out


def _time_adjust(trades):
    out = trades.copy()
    group_med = out.groupby(["_complex", "_size"])["_price"].transform("median")
    out["_group_ratio"] = out["_price"] / group_med
    month_index = out.groupby("_month")["_group_ratio"].median().sort_index()
    month_index = month_index.rolling(3, min_periods=1, center=True).median()
    latest = month_index.iloc[-1] if len(month_index) else 1.0
    out["_time_index"] = out["_month"].map(month_index).fillna(1.0)
    out["_adjusted_price"] = out["_price"] * latest / out["_time_index"]
    return out


def _floor_factors(adjusted):
    valid = adjusted[adjusted["_band"].isin(["first", "low", "middle", "high", "top"])].copy()
    if valid.empty:
        return {}
    group_mid = valid.groupby(["_complex", "_size"])["_adjusted_price"].transform("median")
    valid["_residual"] = valid["_adjusted_price"] / group_mid
    middle = valid.loc[valid["_band"] == "middle", "_residual"].median()
    middle = middle if pd.notna(middle) and middle > 0 else 1.0
    result = {}
    for band, rows in valid.groupby("_band"):
        raw = rows["_residual"].median() / middle
        n = len(rows)
        result[band] = 1.0 + (raw - 1.0) * n / (n + 12.0)
    result["middle"] = 1.0
    return result


def _confidence(n, floor_n, recent_n):
    score = min(n / 12, 1) * 0.5 + min(floor_n / 25, 1) * 0.25 + min(recent_n / 4, 1) * 0.25
    if score >= 0.72:
        return "높음"
    if score >= 0.42:
        return "보통"
    return "낮음"


def _days_for_gap(base_days, gap_pct):
    if gap_pct <= -3:
        factor = 0.55
    elif gap_pct <= 0:
        factor = 0.8
    elif gap_pct <= 3:
        factor = 1.15
    elif gap_pct <= 6:
        factor = 1.65
    else:
        factor = 2.4
    low = max(14, int(base_days * factor * 0.75))
    high = max(low + 10, int(base_days * factor * 1.35))
    return f"{low}~{high}일"


def build_prediction(listings, trades, area, complex_name, dong, size, floor, total_floor, asking_price=None, target_sqm=None):
    if trades is None or trades.empty:
        return {"available": False, "message": "거래내역 자료가 없어 가격을 계산할 수 없습니다."}
    adjusted = _time_adjust(trades)
    factors = _floor_factors(adjusted)
    target_complex = normalize_text(complex_name)
    target_size = normalize_size(size)
    target_band = floor_band(floor, total_floor)
    floor_factor = factors.get(target_band, 1.0)

    peer = adjusted[(adjusted["_complex"] == target_complex) & (adjusted["_size"] == target_size)].copy()
    # 국토부 원본은 평형 대신 전용면적을 제공한다. 이름/평형 직접 일치가 없으면
    # 동일 단지군의 전용면적 묶음 중 현재 매물 호가와 가장 가까운 묶음을 선택한다.
    if peer.empty and "_family" in adjusted.columns:
        target_family = complex_family(complex_name)
        family_rows = adjusted[adjusted["_family"] == target_family].copy()
        official_size_rows = family_rows[family_rows["_size"] == target_size].copy()
        if not official_size_rows.empty:
            peer = official_size_rows
        if target_sqm is not None and not family_rows.empty:
            exact_area = family_rows[(family_rows["_sqm"] - float(target_sqm)).abs() <= 0.2].copy()
            if peer.empty and not exact_area.empty:
                peer = exact_area
        listing_peer_for_map = listings[(listings["_complex"] == target_complex) & (listings["_size"] == target_size)]
        asking_median = listing_peer_for_map["_price"].dropna().median()
        recent_cut = family_rows["_date"].max() - pd.DateOffset(years=3) if not family_rows.empty else pd.NaT
        recent_family = family_rows[family_rows["_date"] >= recent_cut] if pd.notna(recent_cut) else family_rows
        if peer.empty and not recent_family.empty and recent_family["_sqm"].notna().any():
            recent_family["_sqm_group"] = recent_family["_sqm"].round(1)
            stats = recent_family.groupby("_sqm_group")["_adjusted_price"].median()
            if pd.notna(asking_median) and len(stats):
                chosen_sqm = (stats - asking_median).abs().idxmin()
            else:
                pnum = parse_number(size) or 0
                ordered = sorted(stats.index)
                chosen_sqm = min(ordered, key=lambda x: abs(x * 0.32 - pnum))
            peer = family_rows[(family_rows["_sqm"].round(1) - chosen_sqm).abs() <= 0.2].copy()
    if peer.empty:
        return {"available": False, "message": "선택한 단지·평형과 일치하는 실거래가 없어 아직 예측할 수 없습니다."}

    peer["_factor"] = peer["_band"].map(factors).fillna(1.0)
    peer["_standard_price"] = peer["_adjusted_price"] / peer["_factor"]
    center = float(peer["_standard_price"].median() * floor_factor)
    dispersion = float((peer["_standard_price"] / peer["_standard_price"].median() - 1).abs().median()) if len(peer) > 1 else 0.06
    spread = min(max(dispersion, 0.025), 0.09)

    listing_peer = listings[(listings["_complex"] == target_complex) & (listings["_size"] == target_size)].copy()
    inventory = int(len(listing_peer))
    complete_cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=35)
    recent_start = complete_cutoff - pd.DateOffset(months=6)
    recent = peer[(peer["_date"] >= recent_start) & (peer["_date"] <= complete_cutoff)]
    monthly_sales = len(recent) / 6.0
    base_days = 180.0 if monthly_sales <= 0 else max(28.0, min(240.0, (max(inventory, 1) / monthly_sales) * 30.4))

    quick = (center * (1 - max(spread, 0.035)), center * (1 - 0.01))
    fair = (center * (1 - 0.015), center * (1 + 0.02))
    target = (center * (1 + 0.025), center * (1 + max(spread, 0.055)))
    quick_days = _days_for_gap(base_days, -3.5)
    fair_days = _days_for_gap(base_days, 1.0)
    target_days = _days_for_gap(base_days, 5.0)
    asking_gap = ((asking_price / center) - 1) * 100 if asking_price else None
    asking_days = _days_for_gap(base_days, asking_gap) if asking_price else None

    floor_n = int((adjusted["_band"] == target_band).sum())
    confidence = _confidence(len(peer), floor_n, len(recent))
    evidence_cols = []
    display = peer.sort_values("_date", ascending=False).head(10).copy()
    display["계약일"] = display["_date"].dt.strftime("%Y-%m-%d")
    display["거래가(억)"] = display["_price"].round(2)
    display["층"] = display["_floor"]
    display["시점보정가(억)"] = display["_adjusted_price"].round(2)
    evidence = display[["계약일", "거래가(억)", "층", "시점보정가(억)"]]

    comp_cols = [c for c in ["단지명", "동", "평형", "층수", "가격"] if c in listing_peer.columns]
    competitors = listing_peer[comp_cols].copy().head(30)

    return {
        "available": True,
        "center_price": center,
        "quick_price": quick,
        "fair_price": fair,
        "target_price": target,
        "quick_days": quick_days,
        "fair_days": fair_days,
        "target_days": target_days,
        "asking_gap_pct": asking_gap,
        "asking_days": asking_days,
        "monthly_sales": monthly_sales,
        "inventory": inventory,
        "sample_count": len(peer),
        "confidence": confidence,
        "floor_factor": floor_factor if target_band in factors else None,
        "evidence": evidence,
        "competitors": competitors,
        "method_note": "동일 단지·평형 실거래를 최근 시점으로 환산한 뒤, 압구정 전체 반복거래에서 추정한 상대층 구간 계수를 적용했습니다. 최근 35일 거래는 신고 지연을 고려해 거래속도 계산에서 제외했습니다.",
    }
