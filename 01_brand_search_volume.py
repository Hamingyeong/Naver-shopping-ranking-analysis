# =============================================================
# 01_brand_search_volume.py
# 네이버 키보드 + 의자 브랜드 검색량 수집
#
# 네이버 검색광고 API를 사용하여 브랜드별 월간 검색량을 수집하고
# brand_fame_log feature를 생성합니다.
#
# 실행 순서:
#   01_brand_search_volume.py
#   → 02_preprocessing.py
#   → 03_modeling.py
#
# 입력:
#   naver_keyboard_merged_final.csv
#   naver_chair_merged_final.csv
#
# 출력:
#   naver_keyboard_with_features.csv
#   naver_chair_with_features.csv
#   keyboard_brand_search_volume.csv
#   chair_brand_search_volume.csv
# =============================================================

import pandas as pd
import numpy as np
import requests
import time
import os
import re
import hmac
import hashlib
import base64

# ============================================================
# 0. 경로 설정  ← 본인 환경에 맞게 수정
# ============================================================
BASE_DIR = r"c:\capstone design"

# ============================================================
# 1. API 인증 정보  ← config.py에서 불러오거나 직접 입력
# ============================================================
# 방법 A: config.py 파일을 만들어서 키를 분리 (권장)
#   config.py 내용:
#     API_KEY     = "실제키"
#     SECRET_KEY  = "실제키"
#     CUSTOMER_ID = "실제키"
try:
    from config import API_KEY, SECRET_KEY, CUSTOMER_ID
except ImportError:
    # 방법 B: 직접 입력
    API_KEY     = "여기에_API_KEY"
    SECRET_KEY  = "여기에_SECRET_KEY"
    CUSTOMER_ID = "여기에_CUSTOMER_ID"

BASE_URL = "https://api.searchad.naver.com"

# ============================================================
# 2. 카테고리별 설정
# ============================================================
CATEGORY_CONFIGS = {
    "keyboard": {
        "data_path":      os.path.join(BASE_DIR, "naver_keyboard_merged_final.csv"),
        "out_volume":     os.path.join(BASE_DIR, "keyboard_brand_search_volume.csv"),
        "out_error":      os.path.join(BASE_DIR, "keyboard_brand_search_volume_errors.csv"),
        "out_final":      os.path.join(BASE_DIR, "naver_keyboard_with_features.csv"),
        "category_kw":    "키보드",
        "valid_cat4":     {"무선키보드", "유선키보드", "키패드"},
        # 서브브랜드 → 모브랜드 / 영문 → 한글 변환
        "brand_alias": {
            "로지텍G":      "로지텍",
            "녹스게이밍기어": "녹스",
            "아우라":        "AULA",
            "AKKO":          "아코",
            "ASUS":          "에이수스",
            "Apple":         "애플",
            "CHERRY":        "체리",
            "COX":           "콕스",
            "DAREU":         "다리우",
            "DELL":          "델",
            "NZXT":          "엔젝스트",
            "QSENN":         "큐센",
            "RAZER":         "레이저",
            "ROCCAT":        "로캣",
            "archon":        "아콘",
        },
        # IP/캐릭터 브랜드 → brand_fame = 0
        "ip_brands": {"잔망루피", "짱구는못말려", "포켓몬", "위글위글"},
    },
    "chair": {
        "data_path":      os.path.join(BASE_DIR, "naver_chair_merged_final.csv"),
        "out_volume":     os.path.join(BASE_DIR, "chair_brand_search_volume.csv"),
        "out_error":      os.path.join(BASE_DIR, "chair_brand_search_volume_errors.csv"),
        "out_final":      os.path.join(BASE_DIR, "naver_chair_with_features.csv"),
        "category_kw":    "의자",
        "valid_cat4":     {"일반의자", "목받침의자", "좌식의자", "스툴"},
        "brand_alias": {
            "린백토리":      "린백",
            "RAZER":         "레이저",
            "SITTINGPOINT":  "시팅포인트",
            "PICCASSO":      "피카소",
        },
        "ip_brands": set(),  # 의자 카테고리는 IP 브랜드 없음
    },
}

NO_BRAND_LABELS = {"노브랜드", "노 브랜드", "", "nan", "None"}

# ============================================================
# 3. API 헬퍼 함수
# ============================================================
def generate_signature(timestamp, method, uri, secret_key):
    message = f"{timestamp}.{method}.{uri}"
    digest  = hmac.new(
        secret_key.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256
    ).digest()
    return base64.b64encode(digest).decode("utf-8")

def get_headers(method, uri):
    ts = str(round(time.time() * 1000))
    return {
        "Content-Type": "application/json; charset=UTF-8",
        "X-Timestamp":  ts,
        "X-API-KEY":    API_KEY,
        "X-Customer":   str(CUSTOMER_ID),
        "X-Signature":  generate_signature(ts, method, uri, SECRET_KEY),
    }

def normalize(text):
    """소문자 + 공백·특수문자 제거 → 매칭 비교용"""
    text = str(text).lower().strip()
    return re.sub(r"[^가-힣a-z0-9]", "", text)

def clean_for_api(text):
    """API 전송용: 한글+영문+숫자만, 공백 제거"""
    text = str(text).strip()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^가-힣a-zA-Z0-9]", "", text)

def parse_volume(x):
    if x is None: return 0
    if isinstance(x, str):
        x = x.strip()
        if x == "< 10": return 5
        x = x.replace(",", "")
    try:    return int(float(x))
    except: return 0

def fetch_volume(brand, category_keyword, sleep_sec=0.8):
    """
    매칭 우선순위:
      1순위) 정규화 exact match
      2순위) 첫 번째 항목 fallback  → [확인필요] 표시
    실패 시 NaN → 0 처리
    """
    uri    = "/keywordstool"
    method = "GET"
    query  = clean_for_api(f"{brand}{category_keyword}")

    base_row = dict(
        brand=brand, query_raw=f"{brand}{category_keyword}",
        query_clean=query, rel_keyword=None,
        pc_volume=np.nan, mobile_volume=np.nan, total_volume=np.nan,
        match_type=None, status_code=None, error_msg=None,
    )

    if not query:
        base_row["match_type"] = "invalid_empty"
        return base_row

    try:
        res = requests.get(
            BASE_URL + uri,
            headers=get_headers(method, uri),
            params={"hintKeywords": query, "showDetail": "1"},
            timeout=15,
        )
        base_row["status_code"] = res.status_code

        if res.status_code != 200:
            base_row["match_type"] = "api_error"
            base_row["error_msg"]  = str(res.text)[:300]
            return base_row

        items = res.json().get("keywordList", [])
        if not items:
            base_row["match_type"] = "no_result"
            return base_row

        # 1순위: 정규화 exact match
        query_norm = normalize(query)
        matched    = next(
            (i for i in items
             if normalize(str(i.get("relKeyword", ""))) == query_norm),
            None
        )
        match_type = "exact"

        # 2순위: 첫 번째 항목 fallback
        if matched is None:
            matched    = items[0]
            match_type = "first_item"

        pc  = parse_volume(matched.get("monthlyPcQcCnt"))
        mob = parse_volume(matched.get("monthlyMobileQcCnt"))
        base_row.update(
            rel_keyword=matched.get("relKeyword"),
            pc_volume=pc, mobile_volume=mob, total_volume=pc + mob,
            match_type=match_type,
        )
        return base_row

    except requests.RequestException as e:
        base_row["match_type"] = "request_exception"
        base_row["error_msg"]  = str(e)[:300]
        return base_row
    finally:
        time.sleep(sleep_sec)

# ============================================================
# 4. 파생변수 생성 함수
# ============================================================
def add_derived_features(df, cfg):
    """brand_fame 합치기 + 공통 파생변수 생성"""
    valid_cat4 = cfg["valid_cat4"]

    # top38, top100, rank_score
    df["top38"]  = (df["organic_rank"] <= 38).astype(int)
    df["top100"] = (df["organic_rank"] <= 100).astype(int)
    df["rank_score"] = df.groupby("keyword")["organic_rank"].transform(
        lambda x: 1.0 - (x - x.min()) / (x.max() - x.min()))

    # has_review, review_missing
    df["has_review"]     = (df["review_count"].fillna(0) > 0).astype(int)
    df["review_missing"] = df["review_count"].isna().astype(int)

    # is_nobrand
    df["is_nobrand"] = (df["brand"] == "노브랜드").astype(int)

    # title_kw_ratio
    def kw_token_overlap(row):
        tokens = set(str(row["keyword"]).split())
        title  = str(row["title"]).lower()
        return sum(1 for t in tokens if t in title) / len(tokens) if tokens else 0.0
    df["title_kw_ratio"] = df.apply(kw_token_overlap, axis=1)

    # category4 정제
    df["category4_missing"] = df["category4"].isna().astype(int)
    df["category4_clean"] = df["category4"].apply(
        lambda x: x if x in valid_cat4 else ("결측" if pd.isna(x) else "기타"))

    return df

# ============================================================
# 5. 카테고리별 수집 실행
# ============================================================
def run_category(cat_name, cfg):
    print(f"\n{'='*60}")
    print(f"  {cat_name.upper()} 브랜드 검색량 수집")
    print(f"{'='*60}")

    # 데이터 로드
    df = pd.read_csv(cfg["data_path"], encoding="utf-8-sig")
    print(f"  shape: {df.shape}")

    brand_alias = cfg["brand_alias"]
    ip_brands   = cfg["ip_brands"]
    cat_kw      = cfg["category_kw"]

    real_brands = sorted(
        b for b in df["brand"].dropna().unique()
        if str(b).strip() not in NO_BRAND_LABELS
    )
    print(f"  실제 브랜드: {len(real_brands)}개 (노브랜드 제외)")
    print(f"\n  {'브랜드':<20s}  {'전송 키워드':<20s}  {'매칭 키워드':<25s}  {'검색량':>8s}  타입")
    print("  " + "-"*88)

    results = []

    for i, brand in enumerate(real_brands, 1):

        # IP 브랜드 → 0 처리
        if brand in ip_brands:
            results.append(dict(
                brand=brand, query_raw="", query_clean="",
                rel_keyword="(IP브랜드)", pc_volume=0, mobile_volume=0,
                total_volume=0, match_type="ip_brand",
                status_code=None, error_msg=None,
            ))
            print(f"  [{i:3d}/{len(real_brands)}] {brand:<20s}  {'':20s}"
                  f"  {'(IP브랜드, 0처리)':<25s}  {'0':>8s}  ip_brand")
            continue

        # 서브브랜드/영문→한글: 검색 키워드만 변경, brand는 원래 값 유지
        query_brand = brand_alias.get(brand, brand)
        row = fetch_volume(query_brand, cat_kw, sleep_sec=0.8)
        row["brand"] = brand  # 원래 브랜드명으로 복원

        results.append(row)

        vol_str = f"{int(row['total_volume']):,}" if pd.notna(row["total_volume"]) else "NaN→0"
        rel_str = str(row["rel_keyword"] or "-")
        flag    = "  [확인필요]" if row["match_type"] == "first_item" else ""

        print(f"  [{i:3d}/{len(real_brands)}] "
              f"{brand:<20s}  {row['query_clean']:<20s}  "
              f"{rel_str:<25s}  {vol_str:>8s}  {row['match_type']}{flag}")

        if i % 20 == 0:
            pd.DataFrame(results).to_csv(
                os.path.join(BASE_DIR, f"{cat_name}_brand_volume_temp.csv"),
                index=False, encoding="utf-8-sig")
            print("    └ 중간 저장 완료")

    # 노브랜드 삽입
    results.append(dict(
        brand="노브랜드", query_raw="", query_clean="",
        rel_keyword="(노브랜드)", pc_volume=0, mobile_volume=0,
        total_volume=0, match_type="no_brand", status_code=None, error_msg=None,
    ))

    # 결과 정리
    vol_df = pd.DataFrame(results)
    vol_df["total_volume_filled"]  = vol_df["total_volume"].fillna(0).astype(int)
    vol_df["pc_volume_filled"]     = vol_df["pc_volume"].fillna(0).astype(int)
    vol_df["mobile_volume_filled"] = vol_df["mobile_volume"].fillna(0).astype(int)
    vol_df["brand_fame_log"]       = np.log1p(vol_df["total_volume_filled"])

    vol_df.sort_values("total_volume_filled", ascending=False).to_csv(
        cfg["out_volume"], index=False, encoding="utf-8-sig")
    vol_df[vol_df["match_type"].isin(
        ["api_error", "request_exception", "invalid_empty"])
    ].to_csv(cfg["out_error"], index=False, encoding="utf-8-sig")

    # 수집 요약
    print(f"\n  match_type 분포:")
    print(vol_df["match_type"].value_counts().to_string())

    first_item_df = vol_df[vol_df["match_type"] == "first_item"]
    if len(first_item_df):
        print(f"\n  [확인필요] first_item {len(first_item_df)}개")
        print(f"  rel_keyword 확인 후 이상하면 {cfg['out_volume']}에서 수동으로 0 수정")
        print(first_item_df[["brand","query_clean","rel_keyword",
                              "total_volume_filled"]].to_string(index=False))

    # 원본 데이터에 brand_fame 합치기 + 파생변수 생성
    brand_map = vol_df.set_index("brand")[["total_volume_filled","brand_fame_log"]]
    df = df.join(brand_map, on="brand", how="left")
    df["total_volume_filled"] = df["total_volume_filled"].fillna(0).astype(int)
    df["brand_fame_log"]      = df["brand_fame_log"].fillna(0.0)

    df = add_derived_features(df, cfg)

    df.to_csv(cfg["out_final"], index=False, encoding="utf-8-sig")
    print(f"\n  최종 데이터 저장: {cfg['out_final']}  {df.shape}")

    return vol_df

# ============================================================
# 6. 검증 로직
# ============================================================
def validate_output(cfg):
    """수집 완료 후 출력 파일 검증"""
    path = cfg["out_final"]
    if not os.path.exists(path):
        print(f"  ⚠️  파일 없음: {path}")
        return False

    df = pd.read_csv(path, encoding="utf-8-sig")
    ok = True

    checks = [
        ("brand_fame_log 존재",   "brand_fame_log" in df.columns),
        ("top38 존재",            "top38" in df.columns),
        ("rank_score 존재",       "rank_score" in df.columns),
        ("brand_fame_log 결측 없음", df["brand_fame_log"].isna().sum() == 0),
        ("top38 비율 5~15%",      0.05 <= df["top38"].mean() <= 0.15),
        ("rank_score 범위 0~1",   df["rank_score"].between(0,1).all()),
        ("category4_clean 존재",  "category4_clean" in df.columns),
    ]

    print(f"\n  검증 결과 ({path}):")
    for name, result in checks:
        status = "✓" if result else "✗"
        print(f"    {status} {name}")
        if not result:
            ok = False

    return ok

# ============================================================
# 7. 실행
# ============================================================
if __name__ == "__main__":

    # API 키 확인
    if API_KEY == "여기에_API_KEY":
        raise ValueError(
            "API_KEY가 설정되지 않았습니다.\n"
            "config.py 파일을 만들거나 코드 내 API_KEY를 직접 입력하세요."
        )

    all_ok = True
    for cat_name, cfg in CATEGORY_CONFIGS.items():
        run_category(cat_name, cfg)
        ok = validate_output(cfg)
        if not ok:
            all_ok = False

    print(f"\n{'='*60}")
    if all_ok:
        print("  모든 검증 통과 ✓  →  02_preprocessing.py 실행 가능")
    else:
        print("  검증 실패 항목 있음 ✗  →  위 오류 확인 후 재실행")
    print(f"{'='*60}")
