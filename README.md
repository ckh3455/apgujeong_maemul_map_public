# apgujeong_maemul_map (public)

압구정 매물 지도 Streamlit 앱(공개용 템플릿)입니다.

## 핵심 동작
- 지도 마커는 기존과 동일하게 **단지명 + 동** 단위로 표시합니다.
- 마커 클릭 후 상단 표는 **(단지명 + 평형) 일치하는 모든 매물(동 무관)**을 **가격 오름차순**으로 출력합니다.
- 거래내역 최신 5건도 **(단지명 + 평형) 일치** 기준으로 조회합니다.
- 빠른 필터는 **20/30/40…평대 → 1/2/3…구역** 순으로 적용하고, 결과는 **가격 오름차순**으로 출력합니다.

## 보안(민감 컬럼 차단)
`app.py` 상단의 Allowlist로 시트에서 읽는 컬럼을 제한합니다.
- LISTING_ALLOW_COLUMNS / LOC_ALLOW_COLUMNS / TRADE_ALLOW_COLUMNS
- 민감한 컬럼이 있다면 Allowlist에서 제거하세요.

## 실행 방법 (로컬)
```bash
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows
# .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Secrets 설정
아래 템플릿을 참고해 `.streamlit/secrets.toml`을 구성합니다.

- `SPREADSHEET_ID`: Google Spreadsheet ID (URL도 가능)
- `GCP_SERVICE_ACCOUNT_JSON`: 서비스 계정 JSON (dict 또는 JSON string)

`secrets.toml.example` 파일을 참고하세요.
