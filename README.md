# 네이버 쇼핑 상위노출 예측 및 패턴 분석

> 데이터마이닝 5팀 | 2025

네이버 쇼핑에서 상품이 상위에 노출되는 패턴을 파악하고 판매자에게 실질적인 인사이트를 제공하는 것을 목적으로 한다. 키보드(전자제품)와 의자(가구) 두 카테고리를 비교 분석하여 상위노출 패턴의 공통점과 차이점을 도출했다.

---

## 분석 개요

| 구분 | 키보드 | 의자 |
|------|--------|------|
| 수집 키워드 | 키보드, 무선 키보드, 게이밍 키보드, 사무용 키보드, 기계식 키보드 | 의자, 게이밍 의자, 바퀴 의자, 사무용 의자, 학생 의자 |
| 전체 상품수 | 2,385개 | 2,368개 |
| top38 비율 | 8.0% | 8.0% |
| 브랜드 수 | 111개 | 171개 |

### 주요 결과

| 지표 | 키보드 (2단계 RF) | 의자 (2단계 RF) |
|------|-----------------|----------------|
| AP | 0.684 | 0.693 |
| F1 | 0.580 | 0.652 |
| 회귀 R² | 0.499 | 0.474 |

- **공통**: review_count, is_lowest_price, tomorrow_delivery가 두 카테고리 모두 상위노출 패턴에서 양(+)의 방향
- **키보드**: 브랜드 인지도(brand_fame_log)가 리뷰 다음으로 중요한 패턴 요인
- **의자**: 가격 경쟁력(is_lowest_price, price)과 평점(rating)이 핵심. 리뷰 제외 시 AP 낙폭이 키보드보다 훨씬 큼 → 리뷰 의존도가 더 강한 카테고리

---

## 프로젝트 구조
.
├── README.md
├── config.py              # API 키 설정 (GitHub에 올리지 않음)
└── modeling_final.py      # 키보드 + 의자 통합 모델링

---

## 실행 환경

```bash
Python 3.9 이상
```

가상환경 사용 권장:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install pandas numpy scikit-learn lightgbm shap matplotlib requests
```

---

## 실행 방법

### 사전 준비

아래 두 파일이 `BASE_DIR`에 있어야 합니다.
naver_keyboard_merged_final.csv   # 키보드 크롤링 데이터
naver_chair_merged_final.csv      # 의자 크롤링 데이터

데이터는 네이버 쇼핑에서 직접 수집한 크롤링 데이터로, 저작권 및 네이버 이용약관에 따라 공개하지 않습니다.

brand_fame_log(브랜드 인지도 feature)는 네이버 검색광고 API로 별도 수집합니다. API 키가 없으면 데이터 내 브랜드 등장 빈도(brand_dataset_count_log)로 자동 대체됩니다.

### 모델링 실행

`modeling_final.py` 상단의 경로를 본인 환경에 맞게 수정하세요.

```python
BASE_DIR = r"c:\capstone design"   # ← 수정
```

```bash
python modeling_final.py
```

키보드와 의자를 순서대로 처리하며, 결과는 `combined_modeling_results/` 폴더에 저장됩니다.

### 출력 파일
combined_modeling_results/
├── all_classification.csv              # 전체 분류 성능 (키보드+의자)
├── all_regression.csv                  # 전체 회귀 성능
├── keyboard_classification.csv         # 키보드 분류 성능
├── keyboard_regression.csv             # 키보드 회귀 성능
├── keyboard_logistic_coef.csv / .png   # 키보드 LR 계수
├── keyboard_ridge_coef.csv / .png      # 키보드 Ridge 계수
├── keyboard_lasso_coef.csv             # 키보드 Lasso 선택 feature
├── keyboard_cls_shap_beeswarm.png      # 키보드 SHAP (리뷰포함)
├── keyboard_cls_no_review_shap_beeswarm.png  # 키보드 SHAP (리뷰제외)
├── keyboard_reg_shap_beeswarm.png      # 키보드 회귀 SHAP
├── keyboard_keyword_results.csv        # 키보드 키워드별 성능
└── chair_*                             # 의자 동일 구조

---

## 분석 방법론

### 모델 구조 (2단계)

top38 비율이 8%로 클래스 불균형이 심각하여 2단계 모델을 설계했다.
전체 데이터
↓
1단계: top100 예측 (positive 21%, 학습 안정)
↓
top100 후보군 500개 (후보군 내 top38 비율 38%)
↓
2단계: top38 재랭킹

### 교차검증

StratifiedKFold(5-fold, shuffle=True)를 적용했다. 같은 키워드 내 중복 상품은 크롤링 방식상 존재하지 않으며, 다른 키워드 간 동일 상품은 키워드마다 rank_score가 다른 독립 관측값으로 처리했다.

### 평가지표

| 지표 | 설명 |
|------|------|
| AP | Average Precision |
| F1 | 정밀도와 재현율의 조화평균 |
| ROC-AUC | 분류 전반적 성능 |
| R² | 회귀 설명력 |

### 주요 feature

| feature | 설명 |
|---------|------|
| review_count_log | 리뷰수 log1p 변환 |
| brand_fame_log | 네이버 검색광고 API 기반 브랜드 인지도 |
| price_competitiveness | 키워드 내 상대적 가격 경쟁력 |
| is_lowest_price | 최저가 여부 |
| tomorrow_delivery | 내일배송 여부 |
| rating_z_score | 키워드 내 상대 평점 |
| title_kw_ratio | 상품명 내 키워드 포함 비율 |
| category4_clean | 세부 카테고리 (one-hot) |

---

## 한계

- **review_count 내생성**: 리뷰수와 상위노출 간 인과관계 구분 불가. 상위노출 상품에서 반복적으로 나타나는 패턴으로 해석
- **R² 한계**: CTR, 체류시간 등 네이버 내부 지표 없이는 구조적 한계 존재 (R² 0.47~0.50)
- **brand_fame_log**: 대표 키워드 기준 검색량이며 proxy 지표로 해석 필요

---

## 참고

- 네이버 검색광고 API: https://developers.naver.com/docs/searchad/api/
- SHAP: https://shap.readthedocs.io
- LightGBM: https://lightgbm.readthedocs.io
