# Naver-shopping-ranking-analysis
네이버 쇼핑 상위 노출 요인 분석
네이버 쇼핑 상위노출 예측 및 패턴 분석
> 데이터마이닝 5팀 | 2025
네이버 쇼핑에서 상품이 상위에 노출되는 패턴을 파악하고, 판매자에게 실질적인 인사이트를 제공하는 것을 목적으로 한다. 키보드(전자제품)와 의자(가구) 두 카테고리를 비교 분석하여 상위노출 패턴의 공통점과 차이점을 도출했다.
---
분석 개요
구분:	키보드	의자
수집: 키워드	키보드, 무선 키보드, 게이밍 키보드, 사무용 키보드, 기계식 키보드	의자, 게이밍 의자, 바퀴 의자, 사무용 의자, 학생 의자
전체 상품수:	2,385개	2,368개
타겟 변수:	top38 (스크롤 없이 노출되는 38위 이내 여부) / rank_score (정규화 순위)	
주요 feature:	리뷰수, 브랜드 인지도, 가격 경쟁력, 평점, 배송 조건 등 34~35개	
---
주요 결과
공통 패턴: 두 카테고리 모두 review_count가 압도적 1위 패턴. is_lowest_price, tomorrow_delivery가 공통적으로 양(+)의 패턴
키보드: 브랜드 인지도(brand_fame_log)가 리뷰 다음으로 중요한 패턴 요인
의자: 가격 경쟁력(is_lowest_price, price)이 핵심. 리뷰 제외 시 AP 낙폭이 키보드보다 훨씬 큼
지표	키보드 (2단계 RF)	의자 (2단계 Logistic)
AP	0.645	0.645
NDCG@38	0.664	0.659
회귀 R²	0.313	0.412
---
프로젝트 구조
```
.
├── README.md
├── brand_fame_pipeline_v4.py       # 키보드 brand_fame 수집 (네이버 검색광고 API)
├── chair_brand_fame_pipeline.py    # 의자 brand_fame 수집
├── preprocessing_final.py          # 전처리 (키보드 기준, 의자도 동일 적용)
└── modeling_combined.py            # 키보드 + 의자 통합 모델링
```
---
실행 환경
```bash
Python 3.9 이상

pip install pandas numpy scikit-learn lightgbm shap matplotlib
```
가상환경 사용 권장:
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install pandas numpy scikit-learn lightgbm shap matplotlib
```
---
실행 순서
1. 데이터 수집
네이버 쇼핑 크롤링 데이터 준비 (별도 크롤러 필요)
```
naver_keyboard_merged_final.csv   # 키보드 크롤링 데이터
naver_chair_merged_final.csv      # 의자 크롤링 데이터
```
2. brand_fame 수집 (네이버 검색광고 API)
API 키 발급: 네이버 검색광고 API
```bash
# 키보드
python brand_fame_pipeline_v4.py

# 의자
python chair_brand_fame_pipeline.py
```
`brand_fame_pipeline_v4.py` 상단의 아래 항목을 본인 키로 교체:
```python
BASE_DIR  = r"c:\capstone design"   # 데이터 폴더 경로
API_KEY     = "여기에_API_KEY"
SECRET_KEY  = "여기에_SECRET_KEY"
CUSTOMER_ID = "여기에_CUSTOMER_ID"
```
출력 파일:
```
naver_keyboard_with_features.csv   # brand_fame_log 포함 키보드 데이터
naver_chair_with_features.csv      # brand_fame_log 포함 의자 데이터
brand_search_volume.csv            # 브랜드별 검색량
```
3. 전처리 (키보드)
```bash
python preprocessing_final.py
```
출력 파일:
```
naver_keyboard_preprocessed.csv
```
> 의자 전처리는 `modeling_combined.py` 내부에 통합되어 있어 별도 실행 불필요
4. 통합 모델링 (키보드 + 의자)
```bash
python modeling_combined.py
```
`modeling_combined.py` 상단의 경로 수정:
```python
BASE_DIR = r"c:\capstone design"   # 본인 환경에 맞게 수정
```
출력 폴더 `combined_modeling_results/`:
```
all_classification.csv              # 전체 분류 성능 (키보드+의자)
all_regression.csv                  # 전체 회귀 성능
keyboard_classification.csv         # 키보드 분류 성능 (랭킹지표 포함)
keyboard_regression.csv             # 키보드 회귀 성능
keyboard_logistic_coef.csv/png      # 키보드 LR 계수
keyboard_ridge_coef.csv/png         # 키보드 Ridge 계수
keyboard_lasso_coef.csv             # 키보드 Lasso 선택 feature
keyboard_cls_shap_beeswarm.png      # 키보드 SHAP (리뷰포함)
keyboard_cls_no_review_shap_beeswarm.png  # 키보드 SHAP (리뷰제외)
keyboard_reg_shap_beeswarm.png      # 키보드 회귀 SHAP
keyboard_keyword_results.csv        # 키보드 키워드별 성능
chair_*                             # 의자 동일 구조
```
---
분석 방법론
모델 구조 (2단계)
top38 비율이 8%로 클래스 불균형이 심각하여 2단계 모델을 설계했다.
```
전체 2,385개 상품
      ↓
1단계: top100 예측 (positive 21%, 학습 안정)
      ↓
top100 후보군 500개 (후보군 내 top38 비율 38%)
      ↓
2단계: top38 재랭킹
```
교차검증
동일 상품(nv_mid)이 여러 키워드에 중복 등장하므로 GroupKFold(5-fold) 적용. 같은 상품이 학습/평가 데이터에 동시에 포함되지 않도록 처리.
평가지표
지표	설명
AP	Average Precision — 전반적인 분류 품질
P@38	Precision@38 — 모델이 top38로 예측한 38개 중 실제 top38 비율
R@38	Recall@38 — 실제 top38 중 모델이 예측한 비율
NDCG@38	순위 품질 — 상위에 실제 top38이 얼마나 집중됐는지
R²	회귀 설명력
주요 feature
feature	설명
review_count_log	리뷰수 log1p 변환
brand_fame_log	네이버 검색광고 API 기반 브랜드 인지도
price_competitiveness	키워드 내 상대적 가격 경쟁력
is_lowest_price	최저가 여부
tomorrow_delivery	내일배송 여부
rating_z_score	키워드 내 상대 평점
title_kw_ratio	상품명 내 키워드 포함 비율
category4_clean	세부 카테고리 (one-hot)
---
한계
review_count 내생성: 리뷰수와 상위노출 간 인과관계 구분 불가. 상위노출 상품에서 반복적으로 나타나는 패턴으로 해석
R² 한계: CTR, 체류시간 등 네이버 내부 지표 없이는 구조적 한계 존재
brand_fame_log: 대표 키워드 기준 검색량이며 proxy 지표로 해석 필요
---
참고
네이버 검색광고 API: https://developers.naver.com/docs/searchad/api/
SHAP: https://shap.readthedocs.io
LightGBM: https://lightgbm.readthedocs.io
