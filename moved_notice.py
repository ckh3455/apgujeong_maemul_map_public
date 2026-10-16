"""Shows the 'moved to apgujeongone.com' notice and stops the old app."""

import streamlit as st

HOMEPAGE_URL = "https://apgujeongone.com/"


def show_moved_notice():
    try:
        st.set_page_config(page_title="압구정 원 부동산 홈페이지로 이전", page_icon="🏠", layout="centered",
                           initial_sidebar_state="collapsed")
    except Exception:
        pass
    st.markdown(
        f"""
<style>
[data-testid="stSidebar"], [data-testid="collapsedControl"], header, footer {{ display: none !important; }}
.apg-moved {{ max-width: 560px; margin: 12vh auto 0; padding: 40px 28px; text-align: center;
  background: #ffffff; border: 1px solid #dbe3f0; border-radius: 18px;
  box-shadow: 0 12px 32px rgba(29, 62, 139, 0.12); }}
.apg-moved h1 {{ font-size: 1.6rem; line-height: 1.4; color: #1d3e8b; margin: 0 0 14px; }}
.apg-moved p {{ font-size: 1.08rem; line-height: 1.7; color: #334155; margin: 0 0 28px; }}
.apg-moved a.apg-btn {{ display: inline-block; padding: 15px 34px; border-radius: 12px; background: #1d3e8b;
  color: #ffffff !important; font-size: 1.1rem; font-weight: 700; text-decoration: none; }}
.apg-moved a.apg-btn:hover {{ background: #16306d; }}
.apg-moved .apg-url {{ margin-top: 16px; font-size: 0.95rem; color: #64748b; }}
</style>
<div class="apg-moved">
  <h1>압구정 원 부동산 홈페이지로<br>기능이 통합되어 업데이트되었습니다</h1>
  <p>앞으로는 압구정 원 부동산 홈페이지를 이용해 주세요.</p>
  <a class="apg-btn" href="{HOMEPAGE_URL}" target="_blank" rel="noopener">압구정 원 부동산 홈페이지로 이동</a>
  <div class="apg-url">{HOMEPAGE_URL}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()
