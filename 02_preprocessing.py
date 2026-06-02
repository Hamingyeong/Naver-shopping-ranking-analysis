# =============================================================
# 02_preprocessing.py
# 네이버 키보드 데이터 전처리
#
# 실행 순서: 01_brand_search_volume.py → 02_preprocessing.py → 03_modeling.py
# 입력: naver_keyboard_with_features.csv
# 출력: naver_keyboard_preprocessed.csv
#
# 의자 전처리는 03_modeling.py 내부에 통합되어 있어 별도 실행 불필요
# =============================================================

import pandas as pd
import numpy as np
import os

BASE_DIR  = r"c:\capstone design"
DATA_PATH = os.path.join(BASE_DIR, "naver_keyboard_with_features.csv")
OUT_PATH  = os.path.join(BASE_DIR, "naver_keyboard_preprocessed.csv")

print("▶ 데이터 로드:", DATA_PATH)
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print(f"  shape: {df.shape}")

# ============================================================
# 1. 컬럼 제거
# ============================================================
DROP_COLS = [
    "is_ad",               # 전체 0, 상수
    "nv_mid",              # 상품 식별자, 학습 불필요
    "title",               # 텍스트, 모델 직접 투입 불가
    "store",               # 텍스트, 판매자명
    "origin_price",        # price + discount_rate로 복원 가능, 중복
    "maker",               # 결측 多, brand와 중복
    "category1",           # 전체 동일값 (디지털/가전)
    "category2",           # 전체 동일값 (주변기기)
    "category3",           # 전체 동일값 (키보드)
    "category4",           # category4_clean으로 대체
    "top20",               # top38로 대체
    "is_rating_missing",   # 1인 행이 1개뿐, 무의미
    "brand",               # brand_fame_log로 대체
    "total_volume_filled", # brand_fame_log로 대체 (raw값)
    "organic_rank",        # 타겟 변수 생성 원천, data leakage
]

DROP_COLS = [c for c in DROP_COLS if c in df.columns]
df.drop(columns=DROP_COLS, inplace=True)
print(f"\n▶ 컬럼 제거 후: {df.shape}")

# ============================================================
# 2. category4_clean 확인 (01에서 생성됐는지)
# ============================================================
if "category4_clean" not in df.columns:
    valid_cat4 = {"무선키보드", "유선키보드", "키패드"}
    df["category4_missing"] = df["category4"].isna().astype(int)
    df["category4_clean"] = df["category4"].apply(
        lambda x: x if x in valid_cat4 else ("결측" if pd.isna(x) else "기타"))

print("\n▶ category4_clean 분포:")
print(df["category4_clean"].value_counts(dropna=False))

# ============================================================
# 3. one-hot 인코딩
# ============================================================
keyword_dummies = pd.get_dummies(df["keyword"], prefix="kw").astype(int)
df = pd.concat([df.drop(columns=["keyword"]), keyword_dummies], axis=1)
print(f"\n▶ keyword one-hot: {keyword_dummies.columns.tolist()}")

cat4_dummies = pd.get_dummies(df["category4_clean"], prefix="cat4").astype(int)
df = pd.concat([df.drop(columns=["category4_clean"]), cat4_dummies], axis=1)
print(f"▶ category4 one-hot: {cat4_dummies.columns.tolist()}")

# ============================================================
# 4. 타겟 변수 확인
# ============================================================
TARGET_COLS = ["top38", "rank_score"]
missing_targets = [c for c in TARGET_COLS if c not in df.columns]
if missing_targets:
    print(f"\n⚠️  타겟 변수 없음: {missing_targets}")
    print("   01_brand_search_volume.py 먼저 실행하세요.")
else:
    print(f"\n▶ 타겟 변수 확인 ✓")
    print(f"  top38     → 1: {df['top38'].sum()}개 ({df['top38'].mean()*100:.1f}%)")
    print(f"  rank_score → min={df['rank_score'].min():.3f}, max={df['rank_score'].max():.3f}")

# ============================================================
# 5. 결측 최종 확인
# ============================================================
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing):
    print(f"\n⚠️  결측 남아있는 컬럼:")
    print(missing)
else:
    print("\n▶ 결측 없음 ✓")

# ============================================================
# 6. 최종 feature 목록 출력
# ============================================================
feature_cols = [c for c in df.columns if c not in TARGET_COLS]
binary  = [c for c in feature_cols
           if df[c].nunique() == 2 and df[c].dtype in [np.int64, np.float64]]
numeric = [c for c in feature_cols
           if df[c].dtype in [np.float64, np.int64] and c not in binary]

print(f"\n{'='*60}")
print(f"최종 feature ({len(feature_cols)}개)  |  타겟: {TARGET_COLS}")
print(f"{'='*60}")
print(f"\n[연속형 ({len(numeric)}개)]")
for c in numeric:
    print(f"  {c:35s} min={df[c].min():8.2f}  max={df[c].max():8.2f}  mean={df[c].mean():.2f}")
print(f"\n[이진형 ({len(binary)}개)]")
for c in binary:
    print(f"  {c:35s} 1 비율={df[c].mean():.3f}")

# ============================================================
# 7. 저장
# ============================================================
df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"\n▶ 저장 완료: {OUT_PATH}")
print(f"  최종 shape: {df.shape}")

# ============================================================
# 8. 검증
# ============================================================
print("\n▶ 검증")
checks = [
    ("brand_fame_log 존재",   "brand_fame_log" in df.columns),
    ("top38 존재",            "top38" in df.columns),
    ("rank_score 존재",       "rank_score" in df.columns),
    ("결측 없음",              df.isnull().sum().sum() == 0),
    ("top38 비율 5~15%",      0.05 <= df["top38"].mean() <= 0.15),
    ("rank_score 범위 0~1",   df["rank_score"].between(0,1).all()),
    ("keyword one-hot 존재",  any(c.startswith("kw_") for c in df.columns)),
    ("cat4 one-hot 존재",     any(c.startswith("cat4_") for c in df.columns)),
]
all_ok = True
for name, result in checks:
    status = "✓" if result else "✗"
    print(f"  {status} {name}")
    if not result:
        all_ok = False

if all_ok:
    print("\n  모든 검증 통과 ✓  →  03_modeling.py 실행 가능")
else:
    print("\n  검증 실패 항목 있음 ✗  →  위 오류 확인 후 재실행")
