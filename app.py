import pandas as pd
import streamlit as st

from data_loader import load_data
from prediction import (
    build_prediction,
    complex_family,
    floor_from_ho,
    normalize_area,
    normalize_size,
    normalize_text,
    parse_number,
    prepare_listings,
    prepare_trades,
    prepare_units,
)


st.set_page_config(page_title="압구정 매도 예측", page_icon="🏢", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 1.6rem;}
    .hero {padding:1.25rem 1.4rem;border-radius:18px;background:linear-gradient(135deg,#14213d,#1f4e79);color:white;margin-bottom:1rem;}
    .hero h1 {margin:0 0 .35rem 0;font-size:2rem;}
    .hero p {margin:0;color:#e8eef7;}
    .result-card {border:1px solid #dce3ea;border-radius:14px;padding:1rem;background:#fff;min-height:132px;}
    .result-label {font-size:.88rem;color:#667085;}
    .result-value {font-size:1.55rem;font-weight:750;color:#14213d;margin:.2rem 0;}
    .result-note {font-size:.84rem;color:#667085;}
    div[data-testid="stMetric"] {border:1px solid #dce3ea;border-radius:14px;padding:12px 14px;background:white;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>압구정 매도 가능가격·기간 예측</h1>
      <p>구역·동·호수를 선택하고 희망 매도가를 입력하면 매도 가능가격과 예상기간을 계산합니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("계산 기준과 제한사항", expanded=False):
    st.markdown(
        """
        - 실거래는 **계약일** 기준으로 계산합니다.
        - 가격 판단은 중개사가 확인해 정리한 **거래내역** 탭을 사용하며, 기록된 거래는 즉시 반영합니다.
        - 가격은 **거래시점과 층만 보정**합니다. 동 위치와 한강 조망 프리미엄은 현재 반영하지 않습니다.
        - 표본이 부족하면 단일 가격을 단정하지 않고 범위와 낮은 신뢰도로 표시합니다.
        """
    )

try:
    raw_listings, raw_units, raw_trades, source_note = load_data()
except Exception as exc:
    st.error("데이터를 불러오지 못했습니다. Streamlit Secrets와 시트 이름을 확인해 주세요.")
    st.exception(exc)
    st.stop()

listings = prepare_listings(raw_listings)
units = prepare_units(raw_units)
trades = prepare_trades(raw_trades, listings, units)

if units.empty:
    st.warning("공동주택 공시가격 탭에서 구역·동·호수 자료를 읽지 못했습니다.")
    st.stop()

st.subheader("1. 물건 입력")
c1, c2, c3 = st.columns(3)

master_columns = ["_area", "_dong", "_complex", "_family", "단지명"]
input_master = units
area_values = sorted(
    [x for x in input_master["_area"].dropna().astype(str).unique() if x],
    key=lambda x: int(x) if x.isdigit() else 999,
)
with c1:
    selected_area = st.selectbox("구역", area_values, format_func=lambda x: f"{x}구역")

area_rows = input_master[input_master["_area"] == selected_area].copy()
dong_values = sorted(
    [x for x in area_rows["_dong"].dropna().astype(str).unique() if x],
    key=lambda x: int(x) if x.isdigit() else 9999,
)
with c2:
    selected_dong = st.selectbox("동", dong_values, format_func=lambda x: f"{x}동")
unit_rows = area_rows[area_rows["_dong"] == selected_dong].copy()
ho_values = sorted(unit_rows["_ho"].dropna().astype(str).unique().tolist(), key=lambda x: int(x))
with c3:
    selected_ho = st.selectbox("호수", ho_values, index=0, format_func=lambda x: f"{x}호")

selected_unit = unit_rows[unit_rows["_ho"] == selected_ho].iloc[0]
complex_name = str(selected_unit["단지명"])
selected_size = f"{parse_number(selected_unit['평형']):g}평" if parse_number(selected_unit["평형"]) is not None else str(selected_unit["평형"])
floor = int(selected_unit["_floor"])
total_floor = int(selected_unit["_total_floor"])
selected_sqm = selected_unit["_sqm"]

p1, p2 = st.columns([1, 1])
with p1:
    st.markdown("**선택 세대 정보**")
    st.info(f"{complex_name} {selected_size}")
with p2:
    asking_price = st.number_input("희망 매도가(억원)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)

run = st.button("예측 계산", type="primary", use_container_width=True)

if run:
    if not selected_size:
        st.error("평형을 선택해 주세요.")
        st.stop()

    result = build_prediction(
        listings=listings,
        trades=trades,
        area=selected_area,
        complex_name=complex_name,
        dong=selected_dong,
        size=selected_size,
        floor=floor,
        total_floor=total_floor,
        asking_price=(asking_price if asking_price > 0 else None),
        target_sqm=selected_sqm,
    )

    st.divider()
    st.subheader("2. 예측 결과")

    if not result["available"]:
        st.warning(result["message"])
        st.stop()

    cards = st.columns(3)
    values = [
        ("빠른 매도", result["quick_price"], result["quick_days"]),
        ("적정 매도", result["fair_price"], result["fair_days"]),
        ("목표 매도", result["target_price"], result["target_days"]),
    ]
    for col, (label, price, days) in zip(cards, values):
        col.markdown(
            f"<div class='result-card'><div class='result-label'>{label}</div>"
            f"<div class='result-value'>{price[0]:.1f}~{price[1]:.1f}억원</div>"
            f"<div class='result-note'>예상 계약기간 {days}</div></div>",
            unsafe_allow_html=True,
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("층 보정 중심가격", f"{result['center_price']:.1f}억원")
    m2.metric("최근 월평균 거래", f"{result['monthly_sales']:.1f}건")
    m3.metric("현재 경쟁 매물", f"{result['inventory']}건")
    m4.metric("신뢰도", result["confidence"])

    if asking_price > 0:
        gap = result["asking_gap_pct"]
        direction = "높음" if gap > 0 else "낮음"
        st.info(
            f"입력한 희망가격 {asking_price:.1f}억원은 층 보정 중심가격보다 "
            f"{abs(gap):.1f}% {direction}. 이 가격의 예상 계약기간은 **{result['asking_days']}**입니다."
        )

    st.subheader("계산 근거")
    e1, e2, e3 = st.columns(3)
    e1.write(f"**대상:** {selected_area}구역 {complex_name} {selected_dong}동 {selected_ho}호")
    e2.write(f"**층 위치:** {floor}층" + (f" / {total_floor}층" if total_floor else ""))
    e3.write(f"**유사 실거래:** {result['sample_count']}건")

    if result["floor_factor"] is not None:
        st.write(f"층 보정계수: **{result['floor_factor']:.3f}** (중간층 기준 1.000)")
    else:
        st.write("층 보정계수: 표본 부족으로 중립값 1.000 적용")

    st.caption(result["method_note"])
    st.warning("동 위치와 조망에 따른 가격 차이는 아직 반영하지 않았습니다.")

    evidence = result.get("evidence")
    if evidence is not None and not evidence.empty:
        st.subheader("사용된 유사 거래")
        st.dataframe(evidence, use_container_width=True, hide_index=True)

    # 경쟁 매물은 원본 매매물건 목록에서 다시 구성해 가격 오름차순을 확실히 유지한다.
    target_complex = normalize_text(complex_name)
    target_size = normalize_size(selected_size)
    comps = listings[
        (listings["_complex"] == target_complex)
        & (listings["_size"] == target_size)
    ].copy()
    comps = comps.sort_values("_price", ascending=True, na_position="last")

    comp_cols = [
        c for c in ["단지명", "동", "평형", "층수", "가격", "가격이력"]
        if c in comps.columns
    ]
    comps = comps[comp_cols].head(30).copy()
    if "가격이력" in comps.columns:
        comps = comps.rename(columns={"가격이력": "가격변동"})

    if comps is not None and not comps.empty:
        st.subheader("현재 경쟁 매물")
        column_config = {}
        if "가격변동" in comps.columns:
            column_config["가격변동"] = st.column_config.TextColumn(
                "가격변동",
                help="매매물건 목록에서 기록된 가격 변경 이력입니다. 최근 변경부터 표시됩니다.",
                width="large",
            )
        st.dataframe(
            comps,
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
        )

st.divider()
st.caption(f"자료 연결: {source_note} · 본 결과는 중개 판단을 보조하는 추정치이며 감정평가 또는 매매 보증가격이 아닙니다.")
