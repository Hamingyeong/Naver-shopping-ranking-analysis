# =============================================================
# 03_eda_charts.py
# 네이버 키보드 + 의자 EDA 시각화
#
# 입력:
#   data/processed/naver_keyboard_with_features.csv
#   data/processed/naver_chair_with_features.csv
#
# 출력:
#   results/eda_results/heatmap.png
#   results/eda_results/boxplot.png
#   results/eda_results/correlation_bar.png
# =============================================================

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 0. 경로 설정
#    - 파일이 src/ 폴더 안에 있어도, 루트에 있어도 작동하도록 처리
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..")) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
OUT_DIR = os.path.join(ROOT_DIR, "results", "eda_results")
os.makedirs(OUT_DIR, exist_ok=True)

KEY_PATH = os.path.join(DATA_DIR, "naver_keyboard_with_features.csv")
CHAIR_PATH = os.path.join(DATA_DIR, "naver_chair_with_features.csv")

# fallback: 기존 전처리 코드 출력 파일명을 사용하는 경우
KEY_FALLBACK = os.path.join(DATA_DIR, "naver_keyboard_merged_final.csv")
CHAIR_FALLBACK = os.path.join(DATA_DIR, "naver_chair_merged_final.csv")

# ============================================================
# 1. 데이터 로드
# ============================================================
def load_category_data(main_path, fallback_path, label):
    path = main_path if os.path.exists(main_path) else fallback_path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} 입력 파일이 없습니다. 다음 경로 중 하나에 파일을 넣어주세요:\n"
            f"- {main_path}\n- {fallback_path}"
        )

    df = pd.read_csv(path, encoding="utf-8-sig")

    # top38이 없으면 organic_rank 기준으로 생성
    if "top38" not in df.columns:
        if "organic_rank" not in df.columns:
            raise KeyError(f"{label}: top38과 organic_rank가 모두 없어 top38을 만들 수 없습니다.")
        df["top38"] = (df["organic_rank"] <= 38).astype(int)

    # brand_fame_log가 없으면 임시 대체값 생성
    # 실제 최종 데이터에는 네이버 데이터랩 기반 brand_fame_log가 포함되어 있어야 함
    if "brand_fame_log" not in df.columns:
        if "brand" in df.columns:
            bc = df["brand"].value_counts()
            df["brand_fame_log"] = np.log1p(df["brand"].map(bc).fillna(0))
        else:
            df["brand_fame_log"] = 0

    return df

key = load_category_data(KEY_PATH, KEY_FALLBACK, "키보드")
chair = load_category_data(CHAIR_PATH, CHAIR_FALLBACK, "의자")

print(f"키보드: {key.shape}  top38={key['top38'].sum()}개")
print(f"의자:   {chair.shape}  top38={chair['top38'].sum()}개")

# ============================================================
# 2. 공통 설정
# ============================================================
CORR_FEATS = [
    "price", "discount_rate", "rating", "review_count_log",
    "free_delivery", "free_return", "tomorrow_delivery", "today_ship",
    "official_store", "top_seller", "is_lowest_price", "has_option",
    "title_length", "kw_in_title", "has_brand",
    "price_competitiveness", "rating_z_score", "brand_fame_log",
]

BOX_FEATS = [
    "review_count_log", "price_competitiveness", "rating_z_score",
    "brand_fame_log", "discount_rate", "title_length",
]

# ============================================================
# 3. 상관관계 Heatmap
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 10))

last_im = None
for ax, df, name in [(axes[0], key, "Keyboard"), (axes[1], chair, "Chair")]:
    feats = [f for f in CORR_FEATS if f in df.columns]
    corr = df[feats + ["top38"]].corr(numeric_only=True)
    last_im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title(f"Correlation Heatmap — {name}", fontsize=13, pad=10)

    idx = list(corr.columns).index("top38")
    for offset in [-0.5, 0.5]:
        ax.axhline(idx + offset, color="gold", linewidth=2)
        ax.axvline(idx + offset, color="gold", linewidth=2)

if last_im is not None:
    plt.colorbar(last_im, ax=axes, shrink=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("heatmap.png 저장 완료")

# ============================================================
# 4. Box Plot — top38 vs 나머지
# ============================================================
fig, axes = plt.subplots(2, 6, figsize=(20, 8))

for col_i, feat in enumerate(BOX_FEATS):
    for row_i, (df, name, color) in enumerate([
        (key, "Keyboard", "#2E4057"),
        (chair, "Chair", "#e74c3c"),
    ]):
        ax = axes[row_i][col_i]
        if feat not in df.columns:
            ax.set_visible(False)
            continue

        g0 = df[df["top38"] == 0][feat].dropna()
        g1 = df[df["top38"] == 1][feat].dropna()

        bp = ax.boxplot(
            [g0, g1],
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            flierprops=dict(marker="o", markersize=2, alpha=0.3),
        )
        bp["boxes"][0].set_facecolor("#aaaaaa")
        bp["boxes"][1].set_facecolor(color)

        ax.set_title(f"{feat}\n({name})", fontsize=8)
        ax.set_xticklabels(["Others", "Top38"], fontsize=8)
        ax.tick_params(labelsize=7)

plt.suptitle("Feature Distribution: Top38 vs Others", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "boxplot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("boxplot.png 저장 완료")

# ============================================================
# 5. top38 상관관계 Bar Chart
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, df, name in [(axes[0], key, "Keyboard"), (axes[1], chair, "Chair")]:
    feats = [f for f in CORR_FEATS if f in df.columns]
    corr_top38 = df[feats + ["top38"]].corr(numeric_only=True)["top38"].drop("top38")
    corr_top38 = corr_top38.sort_values()
    colors = ["#e74c3c" if v < 0 else "#2E4057" for v in corr_top38]

    ax.barh(corr_top38.index, corr_top38.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with top38")
    ax.set_title(f"Feature Correlation with top38 — {name}", fontsize=12)
    ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "correlation_bar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("correlation_bar.png 저장 완료")

# ============================================================
# 6. 완료
# ============================================================
print(f"\n모든 차트 저장: {OUT_DIR}")
print("  heatmap.png         — 상관관계 heatmap")
print("  boxplot.png         — top38 vs 나머지 box plot")
print("  correlation_bar.png — top38 상관관계 bar chart")
