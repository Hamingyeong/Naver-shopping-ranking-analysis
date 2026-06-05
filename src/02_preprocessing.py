# =============================================================
# 02_preprocessing.py
# 네이버 쇼핑 데이터 전처리 — 키보드 + 의자
#
# 네이버 쇼핑 API로 브랜드/카테고리 정보를 수집하고
# 파생변수를 생성하여 모델링용 데이터를 만든다.
#
# 실행 순서: 01_crawling.py → 02_preprocessing.py → 03_modeling.py
#
# 입력:
#   {BASE_DIR}/naver_keyboard_all.csv  (01_crawling.py 출력)
#   {BASE_DIR}/naver_chair_all.csv     (01_crawling.py 출력)
#
# 출력:
#   {BASE_DIR}/naver_keyboard_merged_final.csv
#   {BASE_DIR}/naver_chair_merged_final.csv
# =============================================================

import pandas as pd
import numpy as np
import re
import os
import time
import json
import urllib.request
import urllib.parse

pd.set_option('display.max_columns', None)

# ============================================================
# 0. 경로 설정  ← 본인 환경에 맞게 수정
# ============================================================
BASE_DIR = r"c:\capstone design"

# ============================================================
# 1. 네이버 쇼핑 API 인증 정보  ← 본인 키로 교체
# ============================================================
# 네이버 개발자 센터에서 발급: https://developers.naver.com
# config.py 파일을 만들어서 키를 분리하는 방법 권장:
#   config.py 내용:
#     CLIENT_ID     = "실제_CLIENT_ID"
#     CLIENT_SECRET = "실제_CLIENT_SECRET"
try:
    from config import CLIENT_ID, CLIENT_SECRET
except ImportError:
    CLIENT_ID     = "여기에_CLIENT_ID"
    CLIENT_SECRET = "여기에_CLIENT_SECRET"

# ============================================================
# 2. 카테고리 설정
# ============================================================
CATEGORIES = {
    "keyboard": {
        "input":  os.path.join(BASE_DIR, "naver_keyboard_all.csv"),
        "output": os.path.join(BASE_DIR, "naver_keyboard_merged_final.csv"),
        "label":  "키보드",
    },
    "chair": {
        "input":  os.path.join(BASE_DIR, "naver_chair_all.csv"),
        "output": os.path.join(BASE_DIR, "naver_chair_merged_final.csv"),
        "label":  "의자",
    },
}

# ============================================================
# 3. 네이버 쇼핑 API 함수 (팀원 코드 원본 유지)
# ============================================================
def get_naver_api_info_final(scraped_title, scraped_store, scraped_price, scraped_origin_price, nv_mid):
    """
    100개 검색 + 중고/직구/렌탈 제외 + ID/상점명/원가/할인가 3중 크로스 체크
    """
    cleaned_title = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', '', str(scraped_title))
    cleaned_title = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', cleaned_title)
    query_text = " ".join(cleaned_title.split()[:5])

    if not query_text:
        return None

    encText = urllib.parse.quote(query_text)
    url = f"https://openapi.naver.com/v1/search/shop?query={encText}&display=100&sort=sim&exclude=used:rental:cbshop"

    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", CLIENT_ID)
    request.add_header("X-Naver-Client-Secret", CLIENT_SECRET)

    try:
        response = urllib.request.urlopen(request, timeout=5)
        if response.getcode() == 200:
            data = json.loads(response.read().decode('utf-8'))
            items = data.get('items', [])

            clean_scraped_store = str(scraped_store).replace(" ", "")
            target_price = int(scraped_price)

            try:
                target_origin_price = int(float(scraped_origin_price))
            except:
                target_origin_price = target_price

            for item in items:
                api_product_id = str(item.get('productId', ''))
                api_mall  = str(item.get('mallName', '')).replace(" ", "")
                api_price = int(item.get('lprice', 0)) if item.get('lprice') else 0

                if api_product_id == str(nv_mid):
                    return extract_info(item)

                is_price_match = (
                    abs(api_price - target_price) <= (target_price * 0.1) or
                    abs(api_price - target_origin_price) <= (target_origin_price * 0.1)
                )
                is_store_match = (clean_scraped_store in api_mall) or (api_mall in clean_scraped_store)

                if is_store_match and is_price_match:
                    return extract_info(item)

                if api_mall == "네이버" and is_price_match:
                    first_word = query_text.split()[0] if query_text.split() else ""
                    if first_word and first_word in item.get('title', ''):
                        return extract_info(item)

            return None
    except Exception:
        return None


def extract_info(item):
    return {
        'brand':     item.get('brand', ''),
        'maker':     item.get('maker', ''),
        'category1': item.get('category1', ''),
        'category2': item.get('category2', ''),
        'category3': item.get('category3', ''),
        'category4': item.get('category4', ''),
    }

# ============================================================
# 4. 전처리 함수 (팀원 코드 원본 유지 + top20 → top38 추가)
# ============================================================
def preprocess(df):
    """결측치 처리 및 파생변수 생성"""

    # ── 결측치 처리 ──────────────────────────────────────────
    df['origin_price']  = df['origin_price'].fillna(df['price'])
    df['discount_rate'] = df['discount_rate'].fillna(0)
    df['brand']         = df['brand'].replace('', '노브랜드').fillna('노브랜드')
    df['store']         = df['store'].fillna('알수없음')
    df['review_count']  = df['review_count'].fillna(0)

    # 평점 결측 처리
    df['is_rating_missing'] = df['rating'].isnull().astype(int)
    df['rating'] = df.groupby('keyword')['rating'].transform(
        lambda x: x.fillna(x.median()))
    df['rating'] = df['rating'].fillna(df['rating'].median())

    # 무의미 컬럼 삭제
    df = df.drop(columns=['multi_seller_count', 'fetched_at'], errors='ignore')

    # ── 파생변수 생성 ─────────────────────────────────────────
    # 타겟 변수
    df["top20"] = (df["organic_rank"] <= 20).astype(int)   # 원본 유지
    df["top38"] = (df["organic_rank"] <= 38).astype(int)   # 분석용 추가

    # 상품명 길이
    df["title_length"] = df["title"].apply(lambda x: len(str(x)))

    # 키워드가 상품명에 포함되는지
    df["kw_in_title"] = df.apply(
        lambda row: 1 if str(row['keyword']).replace(" ", "") in str(row['title']).replace(" ", "") else 0,
        axis=1
    )

    # 브랜드 유무
    df["has_brand"] = (df['brand'] != '노브랜드').astype(int)

    # 리뷰수 로그 변환
    df["review_count_log"] = np.log1p(df["review_count"])

    # 가격 경쟁력 (키워드 내 상대 가격)
    keyword_price_mean = df.groupby("keyword")["price"].transform("mean")
    df["price_competitiveness"] = (keyword_price_mean - df["price"]) / keyword_price_mean

    # 평점 z-score (키워드 내 상대 평점)
    keyword_rating_mean = df.groupby("keyword")["rating"].transform("mean")
    keyword_rating_std  = df.groupby("keyword")["rating"].transform("std").replace(0, 1)
    df["rating_z_score"] = (df["rating"] - keyword_rating_mean) / keyword_rating_std

    return df

# ============================================================
# 5. API 수집 + 전처리 실행
# ============================================================
def run_api_and_preprocess(cfg):
    label = cfg["label"]
    print(f"\n{'='*60}")
    print(f"  {label} 전처리 시작")
    print(f"{'='*60}")

    if not os.path.exists(cfg["input"]):
        print(f"  ⚠️  입력 파일 없음: {cfg['input']}")
        print(f"     01_crawling.py를 먼저 실행하세요.")
        return

    df_all = pd.read_csv(cfg["input"], encoding="utf-8-sig")
    print(f"  로드: {df_all.shape}")

    # 광고 제거
    df_all = df_all[df_all['is_ad'] == 0].copy()
    df_all.reset_index(drop=True, inplace=True)
    print(f"  광고 제거 후: {len(df_all)}개")

    # API 수집
    if CLIENT_ID == "여기에_CLIENT_ID":
        print("  ⚠️  CLIENT_ID가 설정되지 않았습니다.")
        print("     config.py에 CLIENT_ID, CLIENT_SECRET을 입력하세요.")
        print("     API 없이 전처리만 진행합니다.")
        api_results = [
            {'brand': '', 'maker': '', 'category1': '',
             'category2': '', 'category3': '', 'category4': ''}
            for _ in range(len(df_all))
        ]
    else:
        print(f"  네이버 쇼핑 API 수집 시작...")
        api_results = []
        for idx, row in df_all.iterrows():
            api_info = get_naver_api_info_final(
                row['title'], row['store'],
                row['price'], row['origin_price'], row['nv_mid']
            )
            api_results.append(api_info if api_info else {
                'brand': '', 'maker': '',
                'category1': '', 'category2': '', 'category3': '', 'category4': ''
            })
            time.sleep(0.2)
            if (idx + 1) % 100 == 0:
                print(f"    ... {idx + 1} / {len(df_all)} 완료")

    api_df    = pd.DataFrame(api_results)
    df_merged = pd.concat([df_all.reset_index(drop=True), api_df], axis=1)
    print("  API 데이터 병합 완료")

    # 전처리
    df_merged = preprocess(df_merged)
    print("  전처리 완료")

    # 저장
    df_merged.to_csv(cfg["output"], index=False, encoding="utf-8-sig")
    print(f"  저장: {cfg['output']}  {df_merged.shape}")
    print(f"  top38 비율: {df_merged['top38'].mean()*100:.1f}%")
    print(f"  top20 비율: {df_merged['top20'].mean()*100:.1f}%")

# ============================================================
# 6. 실행
# ============================================================
print("\n" + "="*60)
print("전처리할 카테고리를 선택하세요.")
print("  1: 키보드")
print("  2: 의자")
print("  3: 키보드 + 의자 모두")
choice = input("선택 (1/2/3): ").strip()

if choice in ["1", "3"]:
    run_api_and_preprocess(CATEGORIES["keyboard"])

if choice in ["2", "3"]:
    run_api_and_preprocess(CATEGORIES["chair"])

print("\n전처리 완료")
print("다음 단계: python 03_modeling.py")
