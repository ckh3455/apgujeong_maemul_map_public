# 압구정 매도 가능가격·기간 예측

기존 지도 중심 앱을 구역·동·호수 입력형 매도 예측 앱으로 개편한 Streamlit 프로젝트입니다.

## 주요 기능

- 구역과 동을 선택하고 호수를 입력하면 층 자동 판정
- 동일 단지·평형 거래를 최근 시점으로 환산
- 상대층 구간을 이용한 층 가격 표준화
- 빠른 매도·적정 매도·목표 매도가격 범위
- 현재 경쟁 매물과 최근 완결 거래량으로 예상 계약기간 산출
- 토지거래허가구역의 약 1개월 신고 지연 반영
- 유사 거래, 표본 수, 신뢰도 공개

동 위치와 조망 프리미엄은 현재 계산하지 않으며 화면에도 미반영 사실을 명확히 표시합니다.

## 필요한 Google Sheets 탭

기본 파일(`SPREADSHEET_ID`):

- `매매물건 목록`: 상태, 구역, 단지명, 동, 평형, 층수, 가격
- `거래내역`: 단지명, 평형, 계약일, 거래가격, 동/호 또는 층

거래자료가 별도 파일에 있으면 `TRADE_SPREADSHEET_ID`를 추가합니다.

## Streamlit Secrets

```toml
SPREADSHEET_ID = "매물 시트 ID 또는 URL"
TRADE_SPREADSHEET_ID = "거래 시트 ID 또는 URL" # 선택
LISTING_TAB = "매매물건 목록"                  # 선택
TRADE_TAB = "거래내역"                         # 선택
GCP_SERVICE_ACCOUNT_JSON = '''{ ... }'''
```

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```
