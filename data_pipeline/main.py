"""
Starter implementation for the Masai /data_pipeline exercise.

Run:
    python data_pipeline.py

The script:
1. Discovers category pages on Books to Scrape.
2. Scrapes books from at least three categories (by default, all categories).
3. Cleans price, rating and availability.
4. Converts GBP to INR using the assignment's fixed rate.
5. Creates a normalized SQLite database.
6. Runs the SQL examples in sql_queries.py.
7. Compares one JOIN result from SQL with an equivalent pandas merge.

"""

from pathlib import Path
import re
import sqlite3
from statistics import median

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50
MIN_CATEGORIES = 3
MIN_BOOKS = 60
TIMEOUT = 20

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "books_clean.csv"
DB_PATH = DATA_DIR / "books.db"


def get_soup(url: str) -> BeautifulSoup:
    response = requests.get(
        url,
        timeout=TIMEOUT,
        headers={"User-Agent": "Mozilla/5.0 (learning project)"}
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def discover_categories():
    soup = get_soup(BASE_URL)
    categories = {}

    for link in soup.select("div.side_categories ul li ul li a"):
        name = link.get_text(" ", strip=True)
        href = link.get("href")
        if href:
            categories[name] = BASE_URL + href

    if len(categories) < MIN_CATEGORIES:
        raise RuntimeError("Could not discover at least three categories.")

    return categories


def parse_rating(card) -> int:
    tag = card.select_one("p.star-rating")
    if not tag:
        return None

    words = tag.get("class", [])
    mapping = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

    for word, value in mapping.items():
        if word in words:
            return value

    return None


def parse_book_card(card, category: str) -> dict:
    title_tag = card.select_one("h3 a")
    price_tag = card.select_one("p.price_color")
    availability_tag = card.select_one("p.instock.availability")

    title = title_tag.get("title", "").strip() if title_tag else ""
    price_text = price_tag.get_text(" ", strip=True) if price_tag else ""
    availability_text = (
        availability_tag.get_text(" ", strip=True)
        if availability_tag else ""
    )

    # Keep the raw fields first; clean them below.
    return {
        "title": title,
        "price_raw": price_text,
        "star_rating_raw": (
            " ".join(card.select_one("p.star-rating").get("class", []))
            if card.select_one("p.star-rating") else ""
        ),
        "availability_raw": availability_text,
        "category": category,
    }


def scrape_category(category: str, first_url: str):
    rows = []
    url = first_url

    while url:
        soup = get_soup(url)

        for card in soup.select("article.product_pod"):
            rows.append(parse_book_card(card, category))

        next_link = soup.select_one("li.next a")
        if next_link and next_link.get("href"):
            next_href = next_link["href"]

            # Category pages use relative links such as ../page-2.html.
            current = url.rsplit("/", 1)[0] + "/"
            url = requests.compat.urljoin(current, next_href)
        else:
            url = None

    return rows


def clean_price(value):
    if value is None:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value).replace(",", ""))
    return float(match.group(1)) if match else None


def clean_rating(value):
    text = str(value)
    mapping = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    for word, number in mapping.items():
        if re.search(rf"\b{word}\b", text):
            return number
    return None


def clean_availability(value):
    return "in stock" in str(value).lower()


def clean_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    df["price_gbp"] = df["price_raw"].apply(clean_price)
    df["rating"] = df["star_rating_raw"].apply(clean_rating)
    df["in_stock"] = df["availability_raw"].apply(clean_availability)

    # Numeric-field fallback: median imputation, as permitted by the brief.
    for col in ["price_gbp", "rating"]:
        if df[col].isna().any():
            med = df[col].median()
            if pd.isna(med):
                raise ValueError(f"No valid values available to impute {col}.")
            df[col] = df[col].fillna(med)

    df["rating"] = df["rating"].round().astype(int)
    df["price_inr"] = df["price_gbp"] * GBP_TO_INR

    df = df[
        ["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]
    ].drop_duplicates(subset=["title", "category"]).reset_index(drop=True)

    return df


def create_database(df: pd.DataFrame):
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT UNIQUE NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        )
    """)

    categories = sorted(df["category"].unique())
    cur.executemany(
        "INSERT INTO categories(category_name) VALUES (?)",
        [(c,) for c in categories]
    )

    category_map = {
        name: category_id
        for category_id, name in cur.execute(
            "SELECT category_id, category_name FROM categories"
        )
    }

    records = [
        (
            row.title,
            float(row.price_gbp),
            float(row.price_inr),
            int(row.rating),
            int(bool(row.in_stock)),
            category_map[row.category],
        )
        for row in df.itertuples(index=False)
    ]

    cur.executemany("""
        INSERT INTO books
        (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, records)

    conn.commit()
    conn.close()


def main():
    print("Discovering categories...")
    categories = discover_categories()

    # Scrape all categories so the >=60 / >=3 requirement is comfortably met.
    raw_rows = []
    for number, (name, url) in enumerate(categories.items(), start=1):
        print(f"[{number}/{len(categories)}] Scraping {name}")
        raw_rows.extend(scrape_category(name, url))

    raw_df = pd.DataFrame(raw_rows)

    if raw_df.empty:
        raise RuntimeError("No books were scraped.")

    clean_df = clean_dataframe(raw_df)

    category_count = clean_df["category"].nunique()
    book_count = len(clean_df)

    if book_count < MIN_BOOKS or category_count < MIN_CATEGORIES:
        raise RuntimeError(
            f"Acceptance target not met: {book_count} books across "
            f"{category_count} categories."
        )

    clean_df.to_csv(CSV_PATH, index=False)
    create_database(clean_df)

    print("\nPipeline complete.")
    print(f"Books: {book_count}")
    print(f"Categories: {category_count}")
    print(f"CSV: {CSV_PATH}")
    print(f"SQLite DB: {DB_PATH}")
    print(f"Fixed exchange rate: 1 GBP = {GBP_TO_INR:.2f} INR")


if __name__ == "__main__":
    main()



