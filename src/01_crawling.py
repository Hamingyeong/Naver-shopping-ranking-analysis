# =============================================================
# 01_crawling.py
# 네이버 쇼핑 크롤링
#
# 실행 순서:
#   1. Chrome 디버그 모드로 열기
#      "C:\Program Files\Google\Chrome\Application\chrome.exe"
#      --remote-debugging-port=9222 --user-data-dir="C:\chrome_debug"
#   2. Chrome에서 네이버 쇼핑 검색결과 열기
#   3. 이 파일 실행 후 터미널에서 엔터 누르면서 스크롤
#
# 출력:
#   {BASE_DIR}/naver_{키워드}.csv     — 키워드별 개별 파일
#   {BASE_DIR}/naver_keyboard_all.csv — 키보드 전체 통합
#   {BASE_DIR}/naver_chair_all.csv    — 의자 전체 통합
# =============================================================

import os, re, time, random, json
from datetime import datetime
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from urllib.parse import quote

# ============================================================
# 0. 경로 설정  ← 본인 환경에 맞게 수정
# ============================================================
BASE_DIR = r"c:\capstone design"
os.makedirs(BASE_DIR, exist_ok=True)

# ============================================================
# 1. 수집할 키워드 목록
# ============================================================
KEYBOARD_KEYWORDS = [
    "키보드",
    "무선 키보드",
    "게이밍 키보드",
    "사무용 키보드",
    "기계식 키보드",
]

CHAIR_KEYWORDS = [
    "의자",
    "게이밍 의자",
    "바퀴 의자",
    "사무용 의자",
    "학생 의자",
]

N_SCROLL = 10  # 스크롤 횟수

# ============================================================
# 2. 드라이버 연결
# ============================================================
def rnd(a=0.8, b=1.8):
    time.sleep(random.uniform(a, b))

def extract_number(text):
    if not text:
        return np.nan
    m = re.search(r"[\d,]+", str(text))
    return int(m.group().replace(",", "")) if m else np.nan

options = Options()
options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
driver = webdriver.Chrome(options=options)
print("연결 완료:", driver.title)

# ============================================================
# 3. 카드 파싱
# ============================================================
def parse_card(card, keyword, rank):
    try:
        link = card.find_element(By.CSS_SELECTOR, "a[data-shp-contents-rank]")
        is_ad = 1 if link.get_attribute("data-shp-contents-grp") == "ad" else 0

        nv_mid = ""
        dtl_raw = link.get_attribute("data-shp-contents-dtl")
        dtl_map = {}
        if dtl_raw:
            dtl = json.loads(dtl_raw)
            dtl_map = {d["key"]: d["value"] for d in dtl}
            nv_mid = dtl_map.get("nv_mid", "") or dtl_map.get("chnl_prod_no", "")

        try:
            title = card.find_element(By.CSS_SELECTOR, "strong[class*='product_card_title']").text.strip()
        except:
            title = dtl_map.get("prod_nm", "")

        try:
            store = card.find_element(By.CSS_SELECTOR, "span[class*='mall_name']").text.strip()
        except:
            store = ""

        try:
            price_text = card.find_element(By.CSS_SELECTOR, "span[class*='priceTag_price__']").text
            price = int(price_text.replace(",", ""))
        except:
            price = int(dtl_map.get("price", 0)) if dtl_map.get("price") else np.nan

        try:
            orig_text = card.find_element(By.CSS_SELECTOR, "span[class*='original_price']").text
            origin_price = extract_number(orig_text)
        except:
            origin_price = np.nan

        try:
            disc = card.find_element(By.CSS_SELECTOR, "span[class*='discount_ratio']").text
            discount_rate = int(re.search(r"\d+", disc).group())
        except:
            discount_rate = 0

        try:
            star_text = card.find_element(By.CSS_SELECTOR, "span[class*='productCardReview_star']").text.strip()
            rating = float(re.search(r"[\d.]+", star_text).group())
        except:
            rating = np.nan

        try:
            rv_els = card.find_elements(By.CSS_SELECTOR, "span[class*='productCardReview_text']")
            review_count = np.nan
            for el in rv_els:
                t = el.text
                if "리뷰" in t:
                    review_count = extract_number(t)
                    break
        except:
            review_count = np.nan

        card_text = card.text
        free_delivery     = 1 if "무료배송" in card_text else 0
        free_return       = 1 if "무료교환반품" in card_text else 0
        tomorrow_delivery = 1 if "내일배송" in card_text or "N내일" in card_text else 0
        today_ship        = 1 if "오늘출발" in card_text else 0
        official_store    = 1 if "공식" in card_text else 0
        top_seller        = 1 if "우수셀러" in card_text else 0
        is_lowest_price   = 1 if "최저가" in card_text else 0

        try:
            opts = card.find_elements(By.CSS_SELECTOR, "div[class*='option'] button, div[class*='swatch'] button")
            has_option = 1 if opts else 0
        except:
            has_option = 0

        try:
            seller_text = card.find_element(By.CSS_SELECTOR, "a[class*='allSellerLink'], button[class*='allSeller'], span[class*='allSeller']").text
            multi_seller_count = extract_number(seller_text)
        except:
            multi_seller_count = np.nan

        return {
            "keyword": keyword, "organic_rank": rank,
            "is_ad": is_ad, "nv_mid": nv_mid,
            "title": title, "store": store,
            "price": price, "origin_price": origin_price,
            "discount_rate": discount_rate,
            "rating": rating, "review_count": review_count,
            "free_delivery": free_delivery, "free_return": free_return,
            "tomorrow_delivery": tomorrow_delivery, "today_ship": today_ship,
            "official_store": official_store, "top_seller": top_seller,
            "is_lowest_price": is_lowest_price, "has_option": has_option,
            "multi_seller_count": multi_seller_count,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        return None

# ============================================================
# 4. 파싱 헬퍼
# ============================================================
def parse_all_cards(keyword, rows):
    cards = driver.find_elements(By.CSS_SELECTOR, "div.basicProductCard_basic_product_card__TdrHT")
    for card in cards:
        try:
            row = parse_card(card, keyword, len(rows) + 1)
            if row is None or row["is_ad"] == 1:
                continue
            rows.append(row)
        except:
            pass
    print(f"  누적 {len(rows)}개 (중복 포함)")

# ============================================================
# 5. 키워드 수집
# ============================================================
def crawl_keyword(keyword):
    print(f"\n{'='*60}")
    print(f"검색어: {keyword}")
    print(f"URL: https://search.shopping.naver.com/ns/search?query={quote(keyword)}")
    input("브라우저에서 검색결과 열고 엔터 ▶ ")

    rows = []

    # 초기 파싱
    print("  [초기] 파싱 중...", end=" ", flush=True)
    parse_all_cards(keyword, rows)

    # 스크롤 N회
    for batch in range(1, N_SCROLL + 1):
        input(f"  [{batch}/{N_SCROLL}] 스크롤 후 엔터 ▶ ")
        print(f"  파싱 중...", end=" ", flush=True)
        parse_all_cards(keyword, rows)

    df = pd.DataFrame(rows)
    before = len(df)
    df = df.drop_duplicates(subset=["nv_mid"], keep="first").reset_index(drop=True)
    df["organic_rank"] = range(1, len(df) + 1)
    print(f"\n수집 완료: {len(df)}개 (중복 {before - len(df)}개 제거)")

    fname = os.path.join(BASE_DIR, f"naver_{keyword.replace(' ', '_')}.csv")
    df.to_csv(fname, index=False, encoding="utf-8-sig")
    print(f"저장: {fname}")
    return df

# ============================================================
# 6. 전체 실행
# ============================================================
print("\n" + "="*60)
print("수집할 카테고리를 선택하세요.")
print("  1: 키보드")
print("  2: 의자")
print("  3: 키보드 + 의자 모두")
choice = input("선택 (1/2/3): ").strip()

all_keyboard_dfs = []
all_chair_dfs    = []

if choice in ["1", "3"]:
    print("\n[키보드 수집 시작]")
    for kw in KEYBOARD_KEYWORDS:
        df_kw = crawl_keyword(kw)
        if len(df_kw) > 0:
            all_keyboard_dfs.append(df_kw)
        rnd(2.0, 3.0)

    if all_keyboard_dfs:
        df_keyboard = pd.concat(all_keyboard_dfs, ignore_index=True)
        out_k = os.path.join(BASE_DIR, "naver_keyboard_all.csv")
        df_keyboard.to_csv(out_k, index=False, encoding="utf-8-sig")
        print(f"\n키보드 전체 저장: {out_k}")
        print(df_keyboard["keyword"].value_counts())

if choice in ["2", "3"]:
    print("\n[의자 수집 시작]")
    for kw in CHAIR_KEYWORDS:
        df_kw = crawl_keyword(kw)
        if len(df_kw) > 0:
            all_chair_dfs.append(df_kw)
        rnd(2.0, 3.0)

    if all_chair_dfs:
        df_chair = pd.concat(all_chair_dfs, ignore_index=True)
        out_c = os.path.join(BASE_DIR, "naver_chair_all.csv")
        df_chair.to_csv(out_c, index=False, encoding="utf-8-sig")
        print(f"\n의자 전체 저장: {out_c}")
        print(df_chair["keyword"].value_counts())

print("\n수집 완료")
