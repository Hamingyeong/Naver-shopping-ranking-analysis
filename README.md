# 네이버 쇼핑 상위노출 요인 분석

## 1. 프로젝트 개요

본 프로젝트는 네이버 쇼핑 검색 결과에서 상품이 상위에 노출되는 패턴을 데이터마이닝 관점에서 분석한 프로젝트이다. 분석 대상은 키보드와 의자 카테고리이며, 개별 상품 자체가 아니라 **상품 × 검색 키워드 단위의 관측치**를 분석 단위로 설정하였다.

동일한 상품이라도 어떤 검색 키워드로 조회되는지에 따라 노출 순위가 달라질 수 있기 때문에, 본 분석에서는 상품 하나를 고정된 대상으로 보기보다 검색 키워드별 노출 결과를 하나의 관측치로 보았다.

본 프로젝트의 목적은 네이버 쇼핑의 실제 랭킹 알고리즘을 복원하는 것이 아니라, 수집된 검색 결과 데이터 안에서 상위노출 상품과 함께 반복적으로 나타나는 feature 패턴을 탐색하는 것이다.

---

## 2. 분석 목적

본 프로젝트의 주요 분석 목적은 다음과 같다.

1. 네이버 쇼핑 검색 결과에서 Top38 및 Top100 상위노출 여부를 예측한다.
2. 상위노출 여부와 관련된 주요 feature 패턴을 파악한다.
3. Top100 후보군을 먼저 선별한 뒤, 그 내부에서 Top38 상품을 재분류하는 2단계 모델 구조를 실험한다.
4. 리뷰수 관련 변수를 제외했을 때 가격, 배송, 브랜드, 제목-키워드 매칭 관련 변수의 중요도가 어떻게 달라지는지 확인한다.
5. 검색 키워드별로 상위노출 예측 성능이 어떻게 달라지는지 비교한다.

---

## 3. 저장소 구성

```text
Naver-shopping-ranking-analysis/
│
├── README.md
├── requirements.txt
│
├── data/
│       ├── naver_keyboard_with_features.csv
│       └── naver_chair_with_features.csv
│
├── src/
│   ├── 01_crawling.py
│   ├── 02_preprocessing.py
│   ├── 03_eda_charts.py
│   └── 04_modeling_final.py
│
└── combined_modeling_results/
└── eda_results/
        ├── heatmap.png
        ├── boxplot.png
        └── correlation_bar.png
```

본 저장소에는 데이터 수집, 전처리, EDA, 모델링에 사용한 코드를 함께 정리하였다.
모델링 결과는 combined_modeling_result 파일에 포함하였고, EDA 시각화 결과는 `eda_results/` 폴더에 포함하였다.

---

## 4. 코드 파일 설명

### `src/01_crawling.py`

네이버 쇼핑 검색 결과를 수집하기 위한 크롤링 코드이다.
검색 키워드별 상품명, 가격, 리뷰 수, 평점, 광고 여부, 순위 정보 등을 수집하는 데 사용하였다.

해당 코드는 수집 당시의 네이버 쇼핑 페이지 구조와 Chrome 실행 환경에 의존한다. 따라서 실행 시점의 페이지 구조, 브라우저 설정, 네트워크 환경에 따라 동일하게 작동하지 않을 수 있다.

### `src/02_preprocessing.py`

크롤링 결과에 네이버 API 및 파생변수를 결합하고, 모델링에 사용할 수 있는 형태로 데이터를 정리하는 전처리 코드이다.

주요 처리 내용은 다음과 같다.

* 네이버 API 기반 브랜드/제조사/카테고리 정보 결합
* 광고 데이터 제거
* 결측치 처리
* 리뷰 수 로그 변환
* 가격 경쟁력 변수 생성
* 평점 z-score 생성
* 제목-키워드 매칭 변수 생성
* 브랜드 여부 변수 생성
* 최종 분석용 CSV 생성

### `src/03_eda_charts.py`

키보드와 의자 데이터를 결합하여 EDA 시각화를 생성하는 코드이다.

주요 산출물은 다음과 같다.

* 주요 feature 간 상관관계 heatmap
* Top38 여부에 따른 feature 분포 비교 boxplot
* Top38 여부와 feature 간 상관관계 bar chart

생성된 EDA 결과 이미지는 `results/eda_results/` 폴더에 저장하였다.

### `src/04_modeling_final.py`

최종 모델링 코드이다.

주요 수행 내용은 다음과 같다.

* 데이터 불러오기
* target 변수 생성
* feature set 구성
* Top100 예측 모델 학습
* Top38 직접 예측 모델 학습
* Top100 후보군 기반 Top38 재분류 모델 학습
* 리뷰수 관련 변수 제외 모델 비교
* rank_score 회귀 분석
* Logistic Regression 계수 해석
* SHAP 기반 feature importance 해석
* 키워드별 모델 성능 비교
* 주요 결과 파일 저장

---

## 5. 데이터 설명

본 저장소에는 크롤링 직후의 원천 데이터가 아니라, 모델링에 바로 사용할 수 있도록 전처리와 feature engineering이 완료된 **분석용 입력 데이터**를 포함하였다.

사용한 입력 파일은 다음과 같다.

```text
data/processed/naver_keyboard_with_features.csv
data/processed/naver_chair_with_features.csv
```

두 파일은 각각 키보드와 의자 카테고리의 네이버 쇼핑 검색 결과 데이터를 기반으로 하며, 동일한 컬럼 구조를 가진다.

주요 변수는 다음과 같다.

* 상품명
* 검색 키워드
* organic rank
* 가격
* 리뷰 수
* 평점
* 브랜드 정보
* 브랜드 관심도
* 배송 관련 정보
* 카테고리 정보
* 제목과 검색 키워드의 매칭 정도
* Top38 여부
* Top100 여부
* 순위 기반 정규화 점수

---

## 6. 원천 데이터 및 재현 관련 안내

본 저장소의 `data/processed/` 폴더에 포함된 CSV 파일은 크롤링 직후의 원천 raw data가 아니라, 전처리 및 feature engineering이 완료된 분석용 입력 데이터이다.

크롤링 및 API 수집 코드는 `src/01_crawling.py`, `src/02_preprocessing.py`에 포함되어 있으나, 실행 시점의 네이버 쇼핑 페이지 구조, Chrome 디버그 모드 설정, 네이버 API 인증 정보, 로컬 파일 경로 등에 따라 동일하게 재현되지 않을 수 있다.

따라서 본 저장소에서 안정적으로 재현 가능한 분석 단계는 `data/processed/`의 전처리 완료 CSV를 기준으로 수행되는 EDA 및 모델링 단계이다.


일부 파생변수는 전체 수집 데이터 기준으로 계산된 값이다. 예를 들어 가격 경쟁력 변수는 동일 검색 키워드 내 평균 가격을 기준으로 해당 상품의 가격이 상대적으로 저렴한지를 나타내도록 생성되었다. 따라서 본 분석은 엄밀한 실시간 예측 시스템이라기보다, 수집된 검색 결과 데이터 안에서 상위노출과 관련된 관찰 패턴을 분석한 탐색적 데이터마이닝 프로젝트로 해석하는 것이 적절하다.

---

## 7. Target 변수

본 분석에서 사용한 주요 target 변수는 다음과 같다.

### 7.1 `top38`

검색 결과에서 organic rank가 38위 이내이면 1, 그렇지 않으면 0으로 설정한 이진 변수이다.

```text
top38 = 1 if organic_rank <= 38
top38 = 0 otherwise
```

Top38은 네이버 쇼핑 검색 결과에서 사용자가 비교적 초기에 확인할 수 있는 상위 영역을 조작적으로 정의한 기준이다.

### 7.2 `top100`

검색 결과에서 organic rank가 100위 이내이면 1, 그렇지 않으면 0으로 설정한 이진 변수이다.

```text
top100 = 1 if organic_rank <= 100
top100 = 0 otherwise
```

Top100은 2단계 모델 구조에서 1단계 후보군 선별 기준으로 사용하였다.

### 7.3 `rank_score`

검색 키워드별 organic rank를 정규화하여 만든 상대적 순위 점수이다.
분류 모델의 결과를 보완하기 위한 회귀 분석 target으로 사용하였다.

---

## 8. 사용한 주요 feature

모델링에는 다음과 같은 feature를 사용하였다.

* 리뷰 수
* 로그 변환 리뷰 수
* 평점
* 가격
* 가격 경쟁력
* 브랜드 여부
* 브랜드 관심도
* 배송 관련 변수
* 카테고리 변수
* 검색 키워드
* 상품명과 검색 키워드의 매칭 정도

단, `organic_rank`, `top38`, `top100`, `rank_score`처럼 정답 생성 또는 평가에 직접 사용되는 변수는 input feature에서 제외하였다.

---

## 9. EDA

EDA에서는 키보드와 의자 데이터를 결합하여 주요 feature의 분포와 상위노출 여부와의 관계를 확인하였다.

EDA 결과는 `src/03_eda_charts.py`를 통해 생성하였으며, 주요 결과 이미지는 `results/eda_results/` 폴더에 포함하였다.

포함된 EDA 결과는 다음과 같다.

```text
results/eda_results/heatmap.png
results/eda_results/boxplot.png
results/eda_results/correlation_bar.png
```

EDA는 모델링에 앞서 feature 간 관계, Top38 여부에 따른 분포 차이, 상위노출 여부와 개별 feature 간 단순 상관관계를 확인하기 위한 탐색적 분석으로 사용하였다.

---

## 10. 분석 방법

본 프로젝트에서는 supervised learning 기반의 분류 및 회귀 모델을 사용하였다.

### 10.1 분류 모델

Top38과 Top100 상위노출 여부를 예측하기 위해 다음 모델을 사용하였다.

* Logistic Regression
* Random Forest
* LightGBM

평가 지표는 다음을 사용하였다.

* Precision
* Recall
* F1-score
* ROC-AUC
* Average Precision
* Precision@38
* Recall@38
* NDCG@38

Top38과 Top100은 양성 클래스의 비율이 낮기 때문에, 클래스 불균형을 고려하여 StratifiedKFold 기반 교차검증을 수행하였다.

---

## 11. 2단계 재분류 구조

Top38은 전체 관측치 중 비율이 낮은 target이므로, 전체 상품에서 Top38을 바로 예측하는 방식뿐만 아니라 2단계 구조도 함께 실험하였다.

2단계 구조는 다음과 같다.

1. 1단계 모델에서 Top100 후보군을 예측한다.
2. 1단계 예측 확률을 기준으로 후보군을 선별한다.
3. 후보군 내부에서 2단계 모델이 Top38 여부를 다시 예측한다.

최종 모델링 코드에서는 후보군 선별 과정의 검증 편향을 줄이기 위해 OOF(out-of-fold) 예측 방식을 사용하였다. 각 관측치는 자기 자신을 학습에 사용하지 않은 1단계 모델의 예측값을 기반으로 후보군에 포함되도록 처리하였다.

---

## 12. 회귀 분석

분류 분석과 함께 `rank_score`를 target으로 하는 회귀 분석도 수행하였다. 회귀 분석은 실제 순위 자체를 정확히 예측하기 위한 목적보다는, 상위노출 여부 분석을 보완하고 feature들이 상대적 순위 점수를 어느 정도 설명하는지 확인하기 위한 보조 분석으로 사용하였다.

사용한 회귀 모델은 다음과 같다.

* Linear Regression
* Ridge Regression
* Random Forest Regressor
* LightGBM Regressor

---

## 13. 실행 방법

### 13.1 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

`requirements.txt`에는 다음 패키지를 포함한다.

```text
pandas
numpy
scikit-learn
lightgbm
shap
matplotlib
seaborn
```

### 13.2 입력 데이터 준비

다음 파일이 `data/processed/` 폴더에 위치해야 한다.

```text
data/processed/naver_keyboard_with_features.csv
data/processed/naver_chair_with_features.csv
```

### 13.3 EDA 실행

```bash
python src/03_eda_charts.py
```

EDA 결과 이미지는 다음 폴더에 저장된다.

```text
results/eda_results/
```

### 13.4 모델링 실행

```bash
python src/04_modeling_final.py
```

모델링 코드를 실행하면 분류 결과, 회귀 결과, 키워드별 결과, feature importance 관련 결과가 생성된다.

단, 본 저장소에는 모델링 결과 파일 전체를 포함하지 않았으며, 필요 시 코드를 실행하여 결과 파일을 생성할 수 있다.

---

## 14. 출력 결과

EDA 결과는 저장소에 포함되어 있으며, 다음 경로에서 확인할 수 있다.

```text
results/eda_results/
```

모델링 실행 시에는 실행 환경에 따라 다음과 같은 결과 파일이 생성될 수 있다.

```text
all_classification.csv
all_regression.csv
keyboard_classification.csv
keyboard_regression.csv
chair_classification.csv
chair_regression.csv
keyboard_keyword_results.csv
chair_keyword_results.csv
```

SHAP 기반 feature importance 이미지와 계수 해석 결과 파일도 함께 생성될 수 있다.

---

## 15. 주요 분석 결과 요약

분석 결과, 키보드와 의자 카테고리 모두에서 리뷰수 관련 변수는 상위노출 여부와 강하게 연결된 패턴을 보였다.

다만 리뷰수는 상위노출의 원인이면서 동시에 상위노출의 결과일 수 있다. 이미 많이 노출된 상품일수록 리뷰가 더 많이 쌓일 가능성이 있기 때문에, 리뷰수 변수는 인과적으로 해석하기 어렵다. 이러한 문제를 고려하여 리뷰수 관련 변수를 제외한 추가 모델도 함께 비교하였다.

리뷰수 관련 변수를 제외했을 때는 카테고리별로 다른 feature가 중요하게 나타났다.

* 키보드 카테고리에서는 가격, 제목-키워드 매칭, 배송 관련 변수가 상대적으로 중요한 패턴을 보였다.
* 의자 카테고리에서는 브랜드 관심도, 가격, 배송, 카테고리 관련 변수가 상대적으로 중요한 패턴을 보였다.

또한 Top100 후보군을 먼저 선별한 뒤 Top38을 재분류하는 2단계 구조는 후보군 내부에서 상위노출 상품을 구분하는 데 유용한 가능성을 보였다.

---

## 16. 한계점

본 분석에는 다음과 같은 한계가 있다.

1. 네이버 쇼핑의 실제 랭킹 알고리즘을 알 수 없기 때문에, 본 분석은 알고리즘 복원이 아니라 관찰 데이터 기반 패턴 분석이다.
2. 리뷰수는 상위노출의 원인이자 결과일 수 있어 내생성 문제가 존재한다.
3. 수집 시점, 검색 환경, 사용자 환경에 따라 네이버 쇼핑 순위는 달라질 수 있다.
4. 크롤링 및 API 수집 코드는 실행 시점의 웹페이지 구조와 API 인증 정보에 따라 동일하게 재현되지 않을 수 있다.
5. 일부 파생변수는 전체 수집 데이터 기준으로 생성되었으므로, 엄밀한 신규 데이터 예측 검증에서는 fold 내부에서 다시 계산하는 방식이 더 적절하다.
6. 본 분석은 상품-키워드 관측치 단위의 상위노출 패턴 분석에 초점을 두었으며, 완전히 새로운 상품에 대한 일반화 성능은 추가 검증이 필요하다.
7. 키워드별 분석은 주요 요인을 직접 비교하기보다는, 키워드별 예측 성능 차이를 확인하는 보조 분석으로 수행하였다.

---

## 17. 추후 개선 방향

추후에는 다음과 같은 방향으로 분석을 개선할 수 있다.

1. 크롤링 및 전처리 코드를 완전 자동화된 파이프라인으로 정리
2. 파생변수를 교차검증 fold 내부에서 재계산하는 방식 적용
3. 더 많은 상품 카테고리와 검색 키워드로 분석 범위 확장
4. 여러 시점의 데이터를 수집하여 시간에 따른 순위 변동 분석
5. 클릭률, 구매 전환율, 광고 노출 여부, 판매량 등 추가 변수 확보
6. 상품 ID 기준 GroupKFold 검증을 통해 신규 상품에 대한 일반화 성능 확인
7. 후보군 선별과 재분류를 통합한 end-to-end ranking model 고도화
8. 키워드별 feature importance 또는 SHAP 분석을 추가하여 검색어별 상위노출 요인 차이 검토

---

## 18. 데이터 출처

본 프로젝트는 다음 자료를 바탕으로 수행하였다.

* 네이버 쇼핑 검색 결과
* 네이버 API 기반 상품 정보
* 네이버 데이터랩 기반 브랜드 관심도 데이터
* 수집된 상품 및 검색 키워드 정보를 바탕으로 생성한 파생 feature
