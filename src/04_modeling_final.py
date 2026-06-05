# =============================================================
# modeling_final.py
# 네이버 쇼핑 상위노출 예측 — 키보드 + 의자 통합 분석
#
# 입력:
#   naver_keyboard_with_features.csv 또는 naver_keyboard_merged_final.csv
#   naver_chair_with_features.csv    또는 naver_chair_merged_final.csv
#
# 출력: combined_modeling_results/ 폴더
#
# [분류]
#   1단계) top100 예측
#   2단계) top100 후보군 → top38 재랭킹
#   베이스라인) top38 직접 예측
#   + review_count 제외 버전
#   모델: Logistic Regression / Random Forest / LightGBM
#   평가: F1, ROC-AUC, AP, P@38, R@38, NDCG@38 (StratifiedKFold 5-fold)
#
# [회귀]
#   타겟: rank_score
#   모델: Linear / Ridge / Lasso / RF / LightGBM
#
# [해석]
#   SHAP (분류: 2단계 LightGBM, 회귀: LightGBM)
#   Logistic Regression 계수
#   Ridge / Lasso 계수
#   키워드별 RF
#
# 교차검증:
#   StratifiedKFold(5-fold) 사용
#   같은 키워드 내 중복 상품은 크롤링 방식상 존재하지 않으며,
#   다른 키워드 간 동일 상품은 rank_score가 다른 독립 관측값으로 처리
# =============================================================

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble        import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 0. 경로 설정  ← 본인 환경에 맞게 수정
# ============================================================
BASE_DIR = r"c:\capstone design"
OUT_DIR  = os.path.join(BASE_DIR, "combined_modeling_results")
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES = {
    "keyboard": {
        "data_path":  os.path.join(BASE_DIR, "naver_keyboard_with_features.csv"),
        "fallback":   os.path.join(BASE_DIR, "naver_keyboard_merged_final.csv"),
        "valid_cat4": {"무선키보드", "유선키보드", "키패드"},
        "cat_keyword":"키보드",
        "label":      "키보드",
    },
    "chair": {
        "data_path":  os.path.join(BASE_DIR, "naver_chair_with_features.csv"),
        "fallback":   os.path.join(BASE_DIR, "naver_chair_merged_final.csv"),
        "valid_cat4": {"일반의자", "목받침의자", "좌식의자", "스툴"},
        "cat_keyword":"의자",
        "label":      "의자",
    },
}

# ============================================================
# 1. 공통 설정
# ============================================================
N_SPLITS = 5
cv_cls   = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
cv_reg   = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
SCORING  = ["f1","roc_auc","average_precision","precision","recall"]


# ── Feature 이름 영문 매핑 (차트용) ────────────────────────
FEATURE_NAME_MAP = {
    # 키워드 (키보드)
    "kw_게이밍 키보드":  "kw_gaming_keyboard",
    "kw_기계식 키보드":  "kw_mechanical_keyboard",
    "kw_무선 키보드":    "kw_wireless_keyboard",
    "kw_사무용 키보드":  "kw_office_keyboard",
    "kw_키보드":         "kw_keyboard",
    # 카테고리 (키보드)
    "cat4_결측":         "cat4_missing",
    "cat4_기타":         "cat4_other",
    "cat4_무선키보드":   "cat4_wireless_keyboard",
    "cat4_유선키보드":   "cat4_wired_keyboard",
    "cat4_키패드":       "cat4_keypad",
    # 키워드 (의자)
    "kw_게이밍 의자":    "kw_gaming_chair",
    "kw_바퀴 의자":      "kw_wheeled_chair",
    "kw_사무용 의자":    "kw_office_chair",
    "kw_의자":           "kw_chair",
    "kw_학생 의자":      "kw_student_chair",
    # 카테고리 (의자)
    "cat4_목받침의자":   "cat4_headrest_chair",
    "cat4_스툴":         "cat4_stool",
    "cat4_일반의자":     "cat4_standard_chair",
    "cat4_좌식의자":     "cat4_floor_chair",
}

def translate_feat_names(feat_names):
    """차트 표시용 한글 feature 이름을 영문으로 변환"""
    return [FEATURE_NAME_MAP.get(f, f) for f in feat_names]

def make_models(spw=11):
    return {
        "Logistic_L2": Pipeline([("scaler",StandardScaler()),
            ("clf",LogisticRegression(C=0.1,class_weight="balanced",
                                      max_iter=2000,random_state=42))]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,max_depth=None,
            max_features="sqrt",min_samples_leaf=2,
            class_weight="balanced",random_state=42,n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300,learning_rate=0.1,
            num_leaves=50,min_child_samples=10,
            reg_alpha=0.1,reg_lambda=0,
            scale_pos_weight=spw,
            random_state=42,n_jobs=-1,verbose=-1),
    }

def eval_cls(name, model, X, y, label, cv=cv_cls):
    scores = cross_validate(model, X, y, cv=cv, scoring=SCORING)
    # 랭킹 지표
    rank_m = ranking_metrics_cv(make_models()[name], X, y, k=38)
    r = {"Model":name,"Target":label,
         "Precision":scores["test_precision"].mean(),
         "Recall":   scores["test_recall"].mean(),
         "F1":       scores["test_f1"].mean(),
         "ROC-AUC":  scores["test_roc_auc"].mean(),
         "AP":       scores["test_average_precision"].mean(),
         "P@38":     rank_m["P@38"],
         "R@38":     rank_m["R@38"],
         "NDCG@38":  rank_m["NDCG@38"]}
    print(f"  {name:20s}| F1={r['F1']:.3f} AUC={r['ROC-AUC']:.3f}"
          f" AP={r['AP']:.3f} P@38={r['P@38']:.3f} NDCG@38={r['NDCG@38']:.3f}")
    return r

def eval_reg(name, model, X, y, label):
    scores = cross_validate(model, X, y, cv=cv_reg,
        scoring=["r2","neg_root_mean_squared_error","neg_mean_absolute_error"])
    r = {"Model":name,"Target":label,
         "R2":  scores["test_r2"].mean(),
         "RMSE":-scores["test_neg_root_mean_squared_error"].mean(),
         "MAE": -scores["test_neg_mean_absolute_error"].mean()}
    print(f"  {name:12s}| R2={r['R2']:.3f}  RMSE={r['RMSE']:.3f}  MAE={r['MAE']:.3f}")
    return r

def ranking_metrics_cv(model, X, y, k=38):
    """Precision@k, Recall@k, NDCG@k — StratifiedKFold 기반"""
    p_list, r_list, ndcg_list = [], [], []
    for tr, te in cv_cls.split(X, y):
        model.fit(X[tr], y[tr])
        proba   = model.predict_proba(X[te])[:, 1]
        top_idx = np.argsort(proba)[::-1][:k]
        hits    = y[te][top_idx].sum()
        p_list.append(hits / k)
        r_list.append(hits / max(y[te].sum(), 1))
        ranked  = y[te][top_idx]
        dcg     = sum(r / np.log2(i+2) for i, r in enumerate(ranked))
        ideal   = sorted(y[te], reverse=True)[:k]
        idcg    = sum(r / np.log2(i+2) for i, r in enumerate(ideal))
        ndcg_list.append(dcg/idcg if idcg > 0 else 0)
    return {f"P@{k}": np.mean(p_list),
            f"R@{k}": np.mean(r_list),
            f"NDCG@{k}": np.mean(ndcg_list)}

def coef_chart(coef_series, title, path, xlabel="Coefficient"):
    top    = coef_series.head(15).sort_values()
    # 한글 feature 이름 영문으로 변환
    top.index = translate_feat_names(list(top.index))
    colors = ["#e74c3c" if v < 0 else "#2E4057" for v in top]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index, top.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(xlabel); ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def shap_charts(sv, X, feat_names, prefix, title_suffix, out_dir):
    # 한글 feature 이름 영문으로 변환
    feat_names_en = translate_feat_names(feat_names)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X, feature_names=feat_names_en, show=False, max_display=15)
    plt.title(f"SHAP Beeswarm — {title_suffix}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{prefix}_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    imp = pd.Series(np.abs(sv).mean(axis=0),
                    index=feat_names).sort_values(ascending=False).head(15)
    imp_en = pd.Series(np.abs(sv).mean(axis=0),
                    index=feat_names_en).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    imp_en[::-1].plot(kind="barh", ax=ax, color="#2E4057")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"SHAP Importance — {title_suffix}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{prefix}_shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    return imp  # 원래 한글 이름으로 반환 (내부 처리용)

# ============================================================
# 2. 전처리 함수
# ============================================================
def prepare_data(cfg):
    path = cfg["data_path"] if os.path.exists(cfg["data_path"]) else cfg["fallback"]
    print(f"  로드: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")

    if "brand_fame_log" not in df.columns:
        bc = df["brand"].value_counts()
        df["brand_fame_log"] = np.log1p(df["brand"].map(bc).fillna(0))
        print("  brand_fame_log: brand_dataset_count_log로 대체")

    df["top38"]  = (df["organic_rank"] <= 38).astype(int)
    df["top100"] = (df["organic_rank"] <= 100).astype(int)
    df["rank_score"] = df.groupby("keyword")["organic_rank"].transform(
        lambda x: 1.0 - (x - x.min()) / (x.max() - x.min()))

    if "has_review"     not in df.columns:
        df["has_review"]     = (df["review_count"].fillna(0) > 0).astype(int)
    if "review_missing" not in df.columns:
        df["review_missing"] = df["review_count"].isna().astype(int)
    if "is_nobrand"     not in df.columns:
        df["is_nobrand"]     = (df["brand"] == "노브랜드").astype(int)
    if "title_kw_ratio" not in df.columns:
        def kw_overlap(row):
            tokens = set(str(row["keyword"]).split())
            title  = str(row["title"]).lower()
            return sum(1 for t in tokens if t in title) / len(tokens) if tokens else 0.0
        df["title_kw_ratio"] = df.apply(kw_overlap, axis=1)

    if "category4_clean" not in df.columns:
        v = cfg["valid_cat4"]
        df["category4_missing"] = df["category4"].isna().astype(int)
        df["category4_clean"] = df["category4"].apply(
            lambda x: x if x in v else ("결측" if pd.isna(x) else "기타"))

    if not any(c.startswith("kw_") and c != "kw_in_title" for c in df.columns):
        df = pd.get_dummies(df, columns=["keyword"], prefix="kw")
    if not any(c.startswith("cat4_") for c in df.columns):
        df = pd.get_dummies(df, columns=["category4_clean"], prefix="cat4")

    DROP   = {"organic_rank","is_ad","nv_mid","title","store","brand","maker",
              "category1","category2","category3","category4","origin_price",
              "top20","is_rating_missing","total_volume_filled","keyword","category4_clean"}
    TARGET = {"top38","top100","rank_score"}
    ALL_FEATS = [c for c in df.columns if c not in DROP | TARGET]
    NO_REVIEW = [f for f in ALL_FEATS
                 if f not in ["review_count","review_count_log","has_review","review_missing"]]
    KW_COLS   = [c for c in ALL_FEATS if c.startswith("kw_") and c != "kw_in_title"]

    for f in ALL_FEATS:
        df[f] = df[f].fillna(0)

    X        = df[ALL_FEATS].values
    X_no_rev = df[NO_REVIEW].values
    y38      = df["top38"].values
    y100     = df["top100"].values
    y_score  = df["rank_score"].values

    print(f"  shape={df.shape}  feature={len(ALL_FEATS)}개"
          f"  top38={y38.sum()}  top100={y100.sum()}")

    return df, X, X_no_rev, y38, y100, y_score, ALL_FEATS, NO_REVIEW, KW_COLS

# ============================================================
# 3. 카테고리별 분석
# ============================================================
all_cls_results = []
all_reg_results = []

for cat_name, cfg in CATEGORIES.items():
    label = cfg["label"]
    print(f"\n{'='*60}")
    print(f"  {label} 분석 시작")
    print(f"{'='*60}")

    df, X, X_no_rev, y38, y100, y_score, ALL_FEATS, NO_REVIEW, KW_COLS = \
        prepare_data(cfg)

    # ── 1단계: top100 ─────────────────────────────────────
    print(f"\n[분류] 1단계: top100")
    cls_results = []
    for n, m in make_models(spw=4).items():
        r = eval_cls(n, m, X, y100, "top100")
        r["category"] = label; cls_results.append(r)

    # ── 베이스라인: top38 직접 ────────────────────────────
    print(f"\n[분류] 베이스라인: top38 직접")
    for n, m in make_models(spw=11).items():
        r = eval_cls(n, m, X, y38, "top38(직접)")
        r["category"] = label; cls_results.append(r)

    # ── 2단계: top38 재랭킹 ───────────────────────────────
    print(f"\n[분류] 2단계: top100 후보 → top38")
    rf1 = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  random_state=42, n_jobs=-1)
    rf1.fit(X, y100)
    top100_idx = np.argsort(rf1.predict_proba(X)[:, 1])[::-1][:500]
    X2  = X[top100_idx];  y2  = y38[top100_idx]
    neg_pos = (y2==0).sum() / max((y2==1).sum(), 1)
    print(f"  후보군 {len(X2)}개 / top38={y2.sum()}개")
    for n, m in make_models(spw=neg_pos).items():
        r = eval_cls(n, m, X2, y2, "top38(2단계)")
        r["category"] = label; cls_results.append(r)

    # ── 2단계: 리뷰 제외 ──────────────────────────────────
    print(f"\n[분류] 2단계: 리뷰제외")
    rf1_nr = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=42, n_jobs=-1)
    rf1_nr.fit(X_no_rev, y100)
    top100_idx_nr = np.argsort(rf1_nr.predict_proba(X_no_rev)[:, 1])[::-1][:500]
    X2_nr = X_no_rev[top100_idx_nr]; y2_nr = y38[top100_idx_nr]
    neg_pos_nr = (y2_nr==0).sum() / max((y2_nr==1).sum(), 1)
    for n, m in make_models(spw=neg_pos_nr).items():
        r = eval_cls(n, m, X2_nr, y2_nr, "top38(2단계,리뷰제외)")
        r["category"] = label; cls_results.append(r)

    cls_df = pd.DataFrame(cls_results).sort_values(
        ["Target","AP"], ascending=[True, False])
    cls_df.to_csv(f"{OUT_DIR}/{cat_name}_classification.csv",
                  index=False, encoding="utf-8-sig")
    all_cls_results.extend(cls_results)

    # ── 회귀 ──────────────────────────────────────────────
    print(f"\n[회귀] rank_score")
    REG_MODELS = {
        "Linear":   Pipeline([("scaler",StandardScaler()),("reg",LinearRegression())]),
        "Ridge":    Pipeline([("scaler",StandardScaler()),("reg",Ridge(alpha=1.0))]),
        "Lasso":    Pipeline([("scaler",StandardScaler()),
                              ("reg",Lasso(alpha=0.01,max_iter=5000))]),
        "RF_Reg":   RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=-1),
        "LGBM_Reg": lgb.LGBMRegressor(n_estimators=300,learning_rate=0.1,
                                       num_leaves=50,random_state=42,n_jobs=-1,verbose=-1),
    }
    reg_results = []
    for n, m in REG_MODELS.items():
        r = eval_reg(n, m, X, y_score, "rank_score")
        r["category"] = label; reg_results.append(r)
    print("  ── 리뷰 제외 ──")
    for n, m in REG_MODELS.items():
        r = eval_reg(n, m, X_no_rev, y_score, "rank_score(리뷰제외)")
        r["category"] = label; reg_results.append(r)

    reg_df = pd.DataFrame(reg_results).sort_values(
        ["Target","R2"], ascending=[True, False])
    reg_df.to_csv(f"{OUT_DIR}/{cat_name}_regression.csv",
                  index=False, encoding="utf-8-sig")
    all_reg_results.extend(reg_results)

    # ── 계수 해석 ─────────────────────────────────────────
    print(f"\n[계수] Logistic / Ridge / Lasso")

    lr = Pipeline([("scaler",StandardScaler()),
        ("clf",LogisticRegression(C=0.1,class_weight="balanced",
                                   max_iter=2000,random_state=42))])
    lr.fit(X, y38)
    coef    = lr.named_steps["clf"].coef_[0]
    lr_coef = pd.DataFrame({"feature":ALL_FEATS,"coefficient":coef,
                             "odds_ratio":np.exp(coef)})
    lr_coef["abs"] = lr_coef["coefficient"].abs()
    lr_coef = lr_coef.sort_values("abs",ascending=False).drop(columns="abs")
    lr_coef.to_csv(f"{OUT_DIR}/{cat_name}_logistic_coef.csv",
                   index=False, encoding="utf-8-sig")
    coef_chart(lr_coef.set_index("feature")["coefficient"],
               f"Logistic Regression 계수 — {label} top38",
               f"{OUT_DIR}/{cat_name}_logistic_coef.png",
               "계수 (양수=상위노출 유리, 음수=불리)")

    ridge = Pipeline([("scaler",StandardScaler()),("reg",Ridge(alpha=1.0))])
    ridge.fit(X, y_score)
    ridge_coef = pd.DataFrame({"feature":ALL_FEATS,
                                "coefficient":ridge.named_steps["reg"].coef_})
    ridge_coef["abs"] = ridge_coef["coefficient"].abs()
    ridge_coef = ridge_coef.sort_values("abs",ascending=False).drop(columns="abs")
    ridge_coef.to_csv(f"{OUT_DIR}/{cat_name}_ridge_coef.csv",
                      index=False, encoding="utf-8-sig")
    coef_chart(ridge_coef.set_index("feature")["coefficient"],
               f"Ridge Regression 계수 — {label} rank_score",
               f"{OUT_DIR}/{cat_name}_ridge_coef.png",
               "계수 (양수=순위 상승, 음수=순위 하락)")

    lasso = Pipeline([("scaler",StandardScaler()),
                      ("reg",Lasso(alpha=0.01,max_iter=5000))])
    lasso.fit(X, y_score)
    lasso_coef = pd.DataFrame({"feature":ALL_FEATS,
                                "coefficient":lasso.named_steps["reg"].coef_})
    lasso_coef["abs"] = lasso_coef["coefficient"].abs()
    lasso_coef = lasso_coef[lasso_coef["abs"]>0].sort_values(
        "abs",ascending=False).drop(columns="abs")
    lasso_coef.to_csv(f"{OUT_DIR}/{cat_name}_lasso_coef.csv",
                      index=False, encoding="utf-8-sig")
    print(f"  Lasso 선택: {len(lasso_coef)}개 / {len(ALL_FEATS)}개")

    # ── SHAP ──────────────────────────────────────────────
    print(f"\n[SHAP] 분류 (2단계 LightGBM)")
    lgb_cls = lgb.LGBMClassifier(n_estimators=300,learning_rate=0.1,num_leaves=50,
        min_child_samples=10,reg_alpha=0.1,reg_lambda=0,
        scale_pos_weight=neg_pos,random_state=42,n_jobs=-1,verbose=-1)
    lgb_cls.fit(X2, y2)
    sv_cls = shap.TreeExplainer(lgb_cls).shap_values(X2)
    imp_cls = shap_charts(sv_cls, X2, ALL_FEATS,
                          f"{cat_name}_cls", f"2단계 LightGBM {label} (top38)", OUT_DIR)
    print(f"  상위 feature:\n{imp_cls.to_string()}")

    print(f"\n[SHAP] 분류 (리뷰제외)")
    lgb_nr = lgb.LGBMClassifier(n_estimators=300,learning_rate=0.1,num_leaves=50,
        min_child_samples=10,reg_alpha=0.1,reg_lambda=0,
        scale_pos_weight=neg_pos_nr,random_state=42,n_jobs=-1,verbose=-1)
    lgb_nr.fit(X2_nr, y2_nr)
    sv_nr = shap.TreeExplainer(lgb_nr).shap_values(X2_nr)
    imp_nr = shap_charts(sv_nr, X2_nr, NO_REVIEW,
                         f"{cat_name}_cls_no_review",
                         f"2단계 LightGBM {label} (top38, 리뷰제외)", OUT_DIR)
    print(f"  상위 feature (리뷰제외):\n{imp_nr.to_string()}")

    print(f"\n[SHAP] 회귀 (LightGBM)")
    lgbm_reg = lgb.LGBMRegressor(n_estimators=300,learning_rate=0.1,num_leaves=50,
        random_state=42,n_jobs=-1,verbose=-1)
    lgbm_reg.fit(X, y_score)
    sv_reg  = shap.TreeExplainer(lgbm_reg).shap_values(X)
    imp_reg = shap_charts(sv_reg, X, ALL_FEATS,
                          f"{cat_name}_reg", f"LightGBM Regressor {label}", OUT_DIR)

    # ── 키워드별 RF ───────────────────────────────────────
    print(f"\n[키워드별 RF] {label}")
    kw_results = []
    for kw in KW_COLS:
        if kw not in df.columns: continue
        sub_X = df[df[kw]==1][ALL_FEATS].values
        sub_y = df[df[kw]==1]["top38"].values
        if sub_y.sum() < 5:
            print(f"  {kw}: top38 부족, 스킵"); continue
        cv_kw = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        sc    = cross_validate(
            RandomForestClassifier(n_estimators=200,min_samples_leaf=2,
                                   class_weight="balanced",random_state=42,n_jobs=-1),
            sub_X, sub_y, cv=cv_kw,
            scoring=["f1","roc_auc","average_precision"])
        r = {"keyword":kw.replace("kw_",""),"n_total":len(sub_y),
             "n_top38":int(sub_y.sum()),"category":label,
             "F1":sc["test_f1"].mean(),"ROC-AUC":sc["test_roc_auc"].mean(),
             "AP":sc["test_average_precision"].mean()}
        kw_results.append(r)
        print(f"  {r['keyword']:15s}  n={r['n_total']:4d}  top38={r['n_top38']:3d}"
              f"  F1={r['F1']:.3f}  AUC={r['ROC-AUC']:.3f}  AP={r['AP']:.3f}")
    if kw_results:
        pd.DataFrame(kw_results).to_csv(
            f"{OUT_DIR}/{cat_name}_keyword_results.csv",
            index=False, encoding="utf-8-sig")

# ============================================================
# 4. 통합 결과 저장
# ============================================================
pd.DataFrame(all_cls_results).sort_values(
    ["category","Target","AP"], ascending=[True,True,False]
).to_csv(f"{OUT_DIR}/all_classification.csv", index=False, encoding="utf-8-sig")

pd.DataFrame(all_reg_results).sort_values(
    ["category","Target","R2"], ascending=[True,True,False]
).to_csv(f"{OUT_DIR}/all_regression.csv", index=False, encoding="utf-8-sig")

# ============================================================
# 5. 검증
# ============================================================
print(f"\n{'='*60}")
print("결과 파일 검증")
print(f"{'='*60}")

required = [
    "all_classification.csv","all_regression.csv",
    "keyboard_classification.csv","keyboard_regression.csv",
    "keyboard_logistic_coef.csv","keyboard_lasso_coef.csv",
    "keyboard_keyword_results.csv",
    "keyboard_cls_shap_beeswarm.png","keyboard_cls_no_review_shap_beeswarm.png",
    "keyboard_reg_shap_beeswarm.png",
    "chair_classification.csv","chair_regression.csv",
    "chair_logistic_coef.csv","chair_lasso_coef.csv",
    "chair_keyword_results.csv",
    "chair_cls_shap_beeswarm.png","chair_cls_no_review_shap_beeswarm.png",
    "chair_reg_shap_beeswarm.png",
]
all_ok = True
for fname in required:
    exists = os.path.exists(f"{OUT_DIR}/{fname}")
    print(f"  {'✓' if exists else '✗'} {fname}")
    if not exists: all_ok = False

try:
    cls_all = pd.read_csv(f"{OUT_DIR}/all_classification.csv")
    checks = [
        ("분류 결과 행수 >= 24",  len(cls_all) >= 24),
        ("키보드 카테고리 존재",   "키보드" in cls_all["category"].values),
        ("의자 카테고리 존재",     "의자"   in cls_all["category"].values),
        ("AP 범위 0~1",           cls_all["AP"].between(0,1).all()),
        ("P@38 컬럼 존재",        "P@38"   in cls_all.columns),
        ("NDCG@38 컬럼 존재",     "NDCG@38" in cls_all.columns),
    ]
    for name, result in checks:
        print(f"  {'✓' if result else '✗'} {name}")
        if not result: all_ok = False
except Exception as e:
    print(f"  ✗ 결과 검증 실패: {e}")
    all_ok = False

print(f"\n{'='*60}")
print(f"  {'모든 검증 통과 ✓' if all_ok else '검증 실패 항목 있음 ✗'}")
print(f"  결과 저장 위치: {OUT_DIR}")
print(f"{'='*60}")
