# =============================================================
# 03_modeling.py
# 네이버 쇼핑 상위노출 예측 — 키보드 + 의자 통합 분석
#
# 입력:
#   naver_keyboard_with_features.csv 또는 naver_keyboard_merged_final.csv
#   naver_chair_with_features.csv    또는 naver_chair_merged_final.csv
#
# 출력: combined_modeling_results/ 폴더
#   {category}_classification.csv   — 분류 성능 (분류지표 + 랭킹지표)
#   {category}_regression.csv       — 회귀 성능
#   {category}_logistic_coef.csv    — LR 계수
#   {category}_ridge_coef.csv       — Ridge 계수
#   {category}_lasso_coef.csv       — Lasso 선택 feature
#   {category}_logistic_coef.png    — LR 계수 차트
#   {category}_ridge_coef.png       — Ridge 계수 차트
#   {category}_shap_cls.png         — 분류 SHAP (리뷰포함)
#   {category}_shap_cls_no_review.png — 분류 SHAP (리뷰제외)
#   {category}_shap_reg.png         — 회귀 SHAP
#   {category}_keyword_results.csv  — 키워드별 성능
# =============================================================

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble        import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

try:
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "NanumGothic"
except:
    pass
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 0. 경로 설정  ← 본인 환경에 맞게 수정
# ============================================================
BASE_DIR = r"c:\capstone design"
OUT_DIR  = os.path.join(BASE_DIR, "combined_modeling_results")
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES = {
    "keyboard": {
        "data_path":    os.path.join(BASE_DIR, "naver_keyboard_with_features.csv"),
        "fallback":     os.path.join(BASE_DIR, "naver_keyboard_merged_final.csv"),
        "valid_cat4":   {"무선키보드", "유선키보드", "키패드"},
        "cat_keyword":  "키보드",
        "label":        "키보드",
    },
    "chair": {
        "data_path":    os.path.join(BASE_DIR, "naver_chair_with_features.csv"),
        "fallback":     os.path.join(BASE_DIR, "naver_chair_merged_final.csv"),
        "valid_cat4":   {"일반의자", "목받침의자", "좌식의자", "스툴"},
        "cat_keyword":  "의자",
        "label":        "의자",
    },
}

# ============================================================
# 1. 공통 헬퍼
# ============================================================
N_SPLITS = 5
CLS_SCORING = ["f1","roc_auc","average_precision","precision","recall"]

def make_cls_models(spw=11):
    return {
        "Logistic_L2": Pipeline([("scaler",StandardScaler()),
            ("clf",LogisticRegression(C=0.1,class_weight="balanced",
                                      max_iter=1000,random_state=42))]),
        "RandomForest": RandomForestClassifier(n_estimators=300,
            class_weight="balanced",random_state=42,n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,
            num_leaves=31,scale_pos_weight=spw,
            random_state=42,n_jobs=-1,verbose=-1),
    }

def ranking_metrics_cv(model, X, y, groups, k=38, n_splits=N_SPLITS):
    """Precision@k, Recall@k, NDCG@k — GroupKFold 기반"""
    gkf = GroupKFold(n_splits=n_splits)
    p_list, r_list, ndcg_list = [], [], []
    for tr, te in gkf.split(X, y, groups):
        model.fit(X[tr], y[tr])
        proba   = model.predict_proba(X[te])[:, 1]
        top_idx = np.argsort(proba)[::-1][:k]
        hits    = y[te][top_idx].sum()
        p_list.append(hits / k)
        r_list.append(hits / max(y[te].sum(), 1))
        ranked  = y[te][top_idx]
        dcg     = sum(r / np.log2(i+2) for i,r in enumerate(ranked))
        ideal   = sorted(y[te], reverse=True)[:k]
        idcg    = sum(r / np.log2(i+2) for i,r in enumerate(ideal))
        ndcg_list.append(dcg/idcg if idcg>0 else 0)
    return {f"P@{k}":np.mean(p_list),
            f"R@{k}":np.mean(r_list),
            f"NDCG@{k}":np.mean(ndcg_list)}

def eval_cls(name, model, X, y, groups, label, spw=11):
    gkf = GroupKFold(n_splits=N_SPLITS)
    scores = cross_validate(model, X, y,
                            cv=gkf.split(X,y,groups),
                            scoring=CLS_SCORING)
    # 랭킹 지표
    rank_m = ranking_metrics_cv(
        make_cls_models(spw)[name], X, y, groups, k=38)
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
          f" AP={r['AP']:.3f} P@38={r['P@38']:.3f}"
          f" R@38={r['R@38']:.3f} NDCG@38={r['NDCG@38']:.3f}")
    return r

def eval_reg(name, model, X, y, groups, label):
    gkf = GroupKFold(n_splits=N_SPLITS)
    scores = cross_validate(model, X, y,
                            cv=gkf.split(X,y,groups),
                            scoring=["r2","neg_root_mean_squared_error",
                                     "neg_mean_absolute_error"])
    r = {"Model":name,"Target":label,
         "R2":  scores["test_r2"].mean(),
         "RMSE":-scores["test_neg_root_mean_squared_error"].mean(),
         "MAE": -scores["test_neg_mean_absolute_error"].mean()}
    print(f"  {name:12s}| R2={r['R2']:.3f}  RMSE={r['RMSE']:.3f}  MAE={r['MAE']:.3f}")
    return r

def coef_chart(coef_series, title, path, xlabel="계수"):
    top    = coef_series.head(15).sort_values()
    colors = ["#e74c3c" if v<0 else "#2E4057" for v in top]
    fig,ax = plt.subplots(figsize=(9,6))
    ax.barh(top.index, top.values, color=colors)
    ax.axvline(0,color="black",linewidth=0.8)
    ax.set_xlabel(xlabel); ax.set_title(title,fontsize=13)
    plt.tight_layout()
    plt.savefig(path,dpi=150,bbox_inches="tight")
    plt.close()

def shap_charts(sv, X, feat_names, prefix, title_suffix, out_dir):
    plt.figure(figsize=(10,8))
    shap.summary_plot(sv,X,feature_names=feat_names,show=False,max_display=15)
    plt.title(f"SHAP Beeswarm — {title_suffix}",fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{prefix}_shap_beeswarm.png",dpi=150,bbox_inches="tight")
    plt.close()
    imp = pd.Series(np.abs(sv).mean(axis=0),
                    index=feat_names).sort_values(ascending=False).head(15)
    fig,ax = plt.subplots(figsize=(9,6))
    imp[::-1].plot(kind="barh",ax=ax,color="#2E4057")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"SHAP Importance — {title_suffix}",fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{prefix}_shap_importance.png",dpi=150,bbox_inches="tight")
    plt.close()
    return imp

# ============================================================
# 2. 전처리 함수
# ============================================================
def prepare_data(cfg):
    path = cfg["data_path"] if os.path.exists(cfg["data_path"]) else cfg["fallback"]
    print(f"  로드: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")

    # brand_fame_log
    if "brand_fame_log" not in df.columns:
        bc = df["brand"].value_counts()
        df["brand_fame_log"] = np.log1p(df["brand"].map(bc).fillna(0))
        print("  brand_fame_log: brand_dataset_count_log로 대체")

    # 타겟
    df["top38"]  = (df["organic_rank"] <= 38).astype(int)
    df["top100"] = (df["organic_rank"] <= 100).astype(int)
    df["rank_score"] = df.groupby("keyword")["organic_rank"].transform(
        lambda x: 1.0-(x-x.min())/(x.max()-x.min()))

    # 파생변수
    if "has_review" not in df.columns:
        df["has_review"] = (df["review_count"].fillna(0)>0).astype(int)
    if "review_missing" not in df.columns:
        df["review_missing"] = df["review_count"].isna().astype(int)
    if "is_nobrand" not in df.columns:
        df["is_nobrand"] = (df["brand"]=="노브랜드").astype(int)
    if "title_kw_ratio" not in df.columns:
        def kw_overlap(row):
            tokens = set(str(row["keyword"]).split())
            title  = str(row["title"]).lower()
            return sum(1 for t in tokens if t in title)/len(tokens) if tokens else 0.0
        df["title_kw_ratio"] = df.apply(kw_overlap, axis=1)

    # category4
    if "category4_clean" not in df.columns:
        v = cfg["valid_cat4"]
        df["category4_missing"] = df["category4"].isna().astype(int)
        df["category4_clean"] = df["category4"].apply(
            lambda x: x if x in v else ("결측" if pd.isna(x) else "기타"))

    # one-hot
    kw_prefix  = f"kw_{cfg['cat_keyword'][0]}"
    cat_prefix = "cat4_"
    has_kw  = any(c.startswith("kw_") and c != "kw_in_title" for c in df.columns
                  if c not in ["kw_in_title"])
    has_cat = any(c.startswith("cat4_") for c in df.columns)
    if not has_kw:
        df = pd.get_dummies(df, columns=["keyword"], prefix="kw")
    if not has_cat:
        df = pd.get_dummies(df, columns=["category4_clean"], prefix="cat4")

    DROP   = {"organic_rank","is_ad","nv_mid","title","store","brand","maker",
              "category1","category2","category3","category4","origin_price",
              "top20","is_rating_missing","total_volume_filled","keyword",
              "category4_clean"}
    TARGET = {"top38","top100","rank_score"}
    ALL_FEATS = [c for c in df.columns if c not in DROP|TARGET]
    NO_REVIEW  = [f for f in ALL_FEATS
                  if f not in ["review_count","review_count_log",
                               "has_review","review_missing"]]
    KW_COLS = [c for c in ALL_FEATS
               if c.startswith("kw_") and c != "kw_in_title"]

    for f in ALL_FEATS:
        df[f] = df[f].fillna(0)

    X        = df[ALL_FEATS].values
    X_no_rev = df[NO_REVIEW].values
    y38      = df["top38"].values
    y100     = df["top100"].values
    y_score  = df["rank_score"].values
    groups   = df["nv_mid"].values

    print(f"  shape={df.shape}  feature={len(ALL_FEATS)}개"
          f"  top38={y38.sum()}  top100={y100.sum()}"
          f"  unique_nv_mid={df['nv_mid'].nunique()}")

    return df, X, X_no_rev, y38, y100, y_score, groups, ALL_FEATS, NO_REVIEW, KW_COLS

# ============================================================
# 3. 카테고리별 분석 실행
# ============================================================
all_cls_results = []
all_reg_results = []

for cat_name, cfg in CATEGORIES.items():
    label = cfg["label"]
    print(f"\n{'='*60}")
    print(f"  {label} 분석 시작")
    print(f"{'='*60}")

    df, X, X_no_rev, y38, y100, y_score, groups, ALL_FEATS, NO_REVIEW, KW_COLS = \
        prepare_data(cfg)

    # ── 1단계: top100 ──────────────────────────────────────
    print(f"\n[분류] 1단계: top100")
    cls_results = []
    for n,m in make_cls_models(spw=4).items():
        r = eval_cls(n,m,X,y100,groups,"top100",spw=4)
        r["category"] = label
        cls_results.append(r)

    # ── 베이스라인: top38 직접 ─────────────────────────────
    print(f"\n[분류] 베이스라인: top38 직접")
    for n,m in make_cls_models(spw=11).items():
        r = eval_cls(n,m,X,y38,groups,"top38(직접)",spw=11)
        r["category"] = label
        cls_results.append(r)

    # ── 2단계: top38 재랭킹 ───────────────────────────────
    print(f"\n[분류] 2단계: top100 후보 → top38")
    rf1 = RandomForestClassifier(n_estimators=300,class_weight="balanced",
                                  random_state=42,n_jobs=-1)
    rf1.fit(X,y100)
    top100_idx = np.argsort(rf1.predict_proba(X)[:,1])[::-1][:500]
    X2  = X[top100_idx]; y2  = y38[top100_idx]; g2  = groups[top100_idx]
    neg_pos = (y2==0).sum()/max((y2==1).sum(),1)
    print(f"  후보군 {len(X2)}개 / top38={y2.sum()}개")
    for n,m in make_cls_models(spw=neg_pos).items():
        r = eval_cls(n,m,X2,y2,g2,"top38(2단계)",spw=neg_pos)
        r["category"] = label
        cls_results.append(r)

    # ── 2단계: 리뷰 제외 ──────────────────────────────────
    print(f"\n[분류] 2단계: 리뷰제외")
    rf1_nr = RandomForestClassifier(n_estimators=300,class_weight="balanced",
                                     random_state=42,n_jobs=-1)
    rf1_nr.fit(X_no_rev,y100)
    top100_idx_nr = np.argsort(rf1_nr.predict_proba(X_no_rev)[:,1])[::-1][:500]
    X2_nr = X_no_rev[top100_idx_nr]
    y2_nr = y38[top100_idx_nr]
    g2_nr = groups[top100_idx_nr]
    neg_pos_nr = (y2_nr==0).sum()/max((y2_nr==1).sum(),1)
    for n,m in make_cls_models(spw=neg_pos_nr).items():
        r = eval_cls(n,m,X2_nr,y2_nr,g2_nr,"top38(2단계,리뷰제외)",spw=neg_pos_nr)
        r["category"] = label
        cls_results.append(r)

    # 분류 저장
    cls_df = pd.DataFrame(cls_results).sort_values(["Target","AP"],ascending=[True,False])
    cls_df.to_csv(f"{OUT_DIR}/{cat_name}_classification.csv",
                  index=False,encoding="utf-8-sig")
    all_cls_results.extend(cls_results)

    # ── 회귀: rank_score ──────────────────────────────────
    print(f"\n[회귀] rank_score")
    REG_MODELS = {
        "Linear":   Pipeline([("scaler",StandardScaler()),("reg",LinearRegression())]),
        "Ridge":    Pipeline([("scaler",StandardScaler()),("reg",Ridge(alpha=1.0))]),
        "Lasso":    Pipeline([("scaler",StandardScaler()),
                              ("reg",Lasso(alpha=0.01,max_iter=5000))]),
        "RF_Reg":   RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=-1),
        "LGBM_Reg": lgb.LGBMRegressor(n_estimators=300,learning_rate=0.05,
                                       num_leaves=31,random_state=42,n_jobs=-1,verbose=-1),
    }
    reg_results = []
    for n,m in REG_MODELS.items():
        r = eval_reg(n,m,X,y_score,groups,"rank_score")
        r["category"] = label; reg_results.append(r)
    print("  ── 리뷰 제외 ──")
    for n,m in REG_MODELS.items():
        r = eval_reg(n,m,X_no_rev,y_score,groups,"rank_score(리뷰제외)")
        r["category"] = label; reg_results.append(r)

    reg_df = pd.DataFrame(reg_results).sort_values(["Target","R2"],ascending=[True,False])
    reg_df.to_csv(f"{OUT_DIR}/{cat_name}_regression.csv",
                  index=False,encoding="utf-8-sig")
    all_reg_results.extend(reg_results)

    # ── 계수 해석 ─────────────────────────────────────────
    print(f"\n[계수] Logistic / Ridge / Lasso")

    lr = Pipeline([("scaler",StandardScaler()),
        ("clf",LogisticRegression(C=0.1,class_weight="balanced",
                                   max_iter=1000,random_state=42))])
    lr.fit(X,y38)
    coef = lr.named_steps["clf"].coef_[0]
    lr_coef = pd.DataFrame({"feature":ALL_FEATS,"coefficient":coef,
                             "odds_ratio":np.exp(coef)})
    lr_coef["abs"] = lr_coef["coefficient"].abs()
    lr_coef = lr_coef.sort_values("abs",ascending=False).drop(columns="abs")
    lr_coef.to_csv(f"{OUT_DIR}/{cat_name}_logistic_coef.csv",
                   index=False,encoding="utf-8-sig")
    coef_chart(lr_coef.set_index("feature")["coefficient"],
               f"Logistic Regression 계수 — {label} top38",
               f"{OUT_DIR}/{cat_name}_logistic_coef.png",
               "계수 (양수=상위노출 유리, 음수=불리)")

    ridge = Pipeline([("scaler",StandardScaler()),("reg",Ridge(alpha=1.0))])
    ridge.fit(X,y_score)
    ridge_coef = pd.DataFrame({"feature":ALL_FEATS,
                                "coefficient":ridge.named_steps["reg"].coef_})
    ridge_coef["abs"] = ridge_coef["coefficient"].abs()
    ridge_coef = ridge_coef.sort_values("abs",ascending=False).drop(columns="abs")
    ridge_coef.to_csv(f"{OUT_DIR}/{cat_name}_ridge_coef.csv",
                      index=False,encoding="utf-8-sig")
    coef_chart(ridge_coef.set_index("feature")["coefficient"],
               f"Ridge Regression 계수 — {label} rank_score",
               f"{OUT_DIR}/{cat_name}_ridge_coef.png",
               "계수 (양수=순위 상승, 음수=순위 하락)")

    lasso = Pipeline([("scaler",StandardScaler()),
                      ("reg",Lasso(alpha=0.01,max_iter=5000))])
    lasso.fit(X,y_score)
    lasso_coef = pd.DataFrame({"feature":ALL_FEATS,
                                "coefficient":lasso.named_steps["reg"].coef_})
    lasso_coef["abs"] = lasso_coef["coefficient"].abs()
    lasso_coef = lasso_coef[lasso_coef["abs"]>0].sort_values(
        "abs",ascending=False).drop(columns="abs")
    lasso_coef.to_csv(f"{OUT_DIR}/{cat_name}_lasso_coef.csv",
                      index=False,encoding="utf-8-sig")
    print(f"  Lasso 선택: {len(lasso_coef)}개 / {len(ALL_FEATS)}개")

    # ── SHAP ──────────────────────────────────────────────
    print(f"\n[SHAP] 분류 (2단계 LightGBM)")
    lgb_cls = lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,
        scale_pos_weight=neg_pos,random_state=42,n_jobs=-1,verbose=-1)
    lgb_cls.fit(X2,y2)
    sv_cls = shap.TreeExplainer(lgb_cls).shap_values(X2)
    imp_cls = shap_charts(sv_cls,X2,ALL_FEATS,
                          f"{cat_name}_cls",f"2단계 LightGBM {label} (top38)",OUT_DIR)
    print(f"  상위 feature:\n{imp_cls.to_string()}")

    print(f"\n[SHAP] 분류 (리뷰제외)")
    lgb_nr = lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,
        scale_pos_weight=neg_pos_nr,random_state=42,n_jobs=-1,verbose=-1)
    lgb_nr.fit(X2_nr,y2_nr)
    sv_nr = shap.TreeExplainer(lgb_nr).shap_values(X2_nr)
    imp_nr = shap_charts(sv_nr,X2_nr,NO_REVIEW,
                         f"{cat_name}_cls_no_review",
                         f"2단계 LightGBM {label} (top38, 리뷰제외)",OUT_DIR)
    print(f"  상위 feature (리뷰제외):\n{imp_nr.to_string()}")

    print(f"\n[SHAP] 회귀 (LightGBM)")
    lgbm_reg = lgb.LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,
        random_state=42,n_jobs=-1,verbose=-1)
    lgbm_reg.fit(X,y_score)
    sv_reg = shap.TreeExplainer(lgbm_reg).shap_values(X)
    imp_reg = shap_charts(sv_reg,X,ALL_FEATS,
                          f"{cat_name}_reg",f"LightGBM Regressor {label}",OUT_DIR)

    # ── 키워드별 RF ────────────────────────────────────────
    print(f"\n[키워드별 RF] {label}")
    kw_results = []
    for kw in KW_COLS:
        if kw not in df.columns: continue
        sub_X = df[df[kw]==1][ALL_FEATS].values
        sub_y = df[df[kw]==1]["top38"].values
        sub_g = df[df[kw]==1]["nv_mid"].values
        if sub_y.sum()<5:
            print(f"  {kw}: top38 부족, 스킵"); continue
        n_splits_kw = min(3, len(set(sub_g)))
        gkf_kw = GroupKFold(n_splits=n_splits_kw)
        sc = cross_validate(
            RandomForestClassifier(n_estimators=200,class_weight="balanced",
                                   random_state=42,n_jobs=-1),
            sub_X,sub_y,cv=gkf_kw.split(sub_X,sub_y,sub_g),
            scoring=["f1","roc_auc","average_precision"])
        r = {"keyword":kw.replace("kw_",""),"n_total":len(sub_y),
             "n_top38":int(sub_y.sum()),"category":label,
             "F1":sc["test_f1"].mean(),
             "ROC-AUC":sc["test_roc_auc"].mean(),
             "AP":sc["test_average_precision"].mean()}
        kw_results.append(r)
        print(f"  {r['keyword']:15s} n={r['n_total']:4d} top38={r['n_top38']:3d}"
              f" F1={r['F1']:.3f} AUC={r['ROC-AUC']:.3f} AP={r['AP']:.3f}")
    if kw_results:
        pd.DataFrame(kw_results).to_csv(
            f"{OUT_DIR}/{cat_name}_keyword_results.csv",
            index=False,encoding="utf-8-sig")

# ============================================================
# 4. 통합 결과 저장
# ============================================================
pd.DataFrame(all_cls_results).sort_values(
    ["category","Target","AP"],ascending=[True,True,False]
).to_csv(f"{OUT_DIR}/all_classification.csv",index=False,encoding="utf-8-sig")

pd.DataFrame(all_reg_results).sort_values(
    ["category","Target","R2"],ascending=[True,True,False]
).to_csv(f"{OUT_DIR}/all_regression.csv",index=False,encoding="utf-8-sig")

print(f"\n{'='*60}")
print(f"모든 결과 저장: {OUT_DIR}")
print(f"{'='*60}")
print("  all_classification.csv        — 전체 분류 성능 (키보드+의자)")
print("  all_regression.csv            — 전체 회귀 성능")
print("  keyboard_classification.csv   — 키보드 분류")
print("  keyboard_regression.csv       — 키보드 회귀")
print("  keyboard_logistic_coef.csv/png")
print("  keyboard_ridge_coef.csv/png")
print("  keyboard_lasso_coef.csv")
print("  keyboard_cls_shap_beeswarm/importance.png")
print("  keyboard_cls_no_review_shap_*.png")
print("  keyboard_reg_shap_*.png")
print("  keyboard_keyword_results.csv")
print("  chair_* (동일 구조)")

# ============================================================
# 검증 로직
# ============================================================
print(f"\n{'='*60}")
print("결과 파일 검증")
print(f"{'='*60}")

import os

required_files = [
    "all_classification.csv",
    "all_regression.csv",
    "keyboard_classification.csv",
    "keyboard_regression.csv",
    "keyboard_logistic_coef.csv",
    "keyboard_lasso_coef.csv",
    "keyboard_keyword_results.csv",
    "keyboard_cls_shap_beeswarm.png",
    "keyboard_cls_no_review_shap_beeswarm.png",
    "keyboard_reg_shap_beeswarm.png",
    "chair_classification.csv",
    "chair_regression.csv",
    "chair_logistic_coef.csv",
    "chair_lasso_coef.csv",
    "chair_keyword_results.csv",
    "chair_cls_shap_beeswarm.png",
    "chair_cls_no_review_shap_beeswarm.png",
    "chair_reg_shap_beeswarm.png",
]

all_ok = True
for fname in required_files:
    fpath = f"{OUT_DIR}/{fname}"
    exists = os.path.exists(fpath)
    status = "✓" if exists else "✗"
    print(f"  {status} {fname}")
    if not exists:
        all_ok = False

# 분류 결과 수치 검증
try:
    cls_all = pd.read_csv(f"{OUT_DIR}/all_classification.csv")
    reg_all = pd.read_csv(f"{OUT_DIR}/all_regression.csv")

    checks = [
        ("분류 결과 행수 >= 24",   len(cls_all) >= 24),
        ("회귀 결과 행수 >= 20",   len(reg_all) >= 20),
        ("키보드 카테고리 존재",    "키보드" in cls_all["category"].values),
        ("의자 카테고리 존재",      "의자" in cls_all["category"].values),
        ("AP 범위 0~1",            cls_all["AP"].between(0,1).all()),
        ("R² 범위 -1~1",           reg_all["R2"].between(-1,1).all()),
        ("P@38 컬럼 존재",         "P@38" in cls_all.columns),
        ("NDCG@38 컬럼 존재",      "NDCG@38" in cls_all.columns),
    ]
    for name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
        if not result:
            all_ok = False
except Exception as e:
    print(f"  ✗ 결과 파일 검증 실패: {e}")
    all_ok = False

print(f"\n{'='*60}")
if all_ok:
    print("  모든 검증 통과 ✓")
else:
    print("  검증 실패 항목 있음 ✗  →  위 오류 확인 후 재실행")
print(f"{'='*60}")
