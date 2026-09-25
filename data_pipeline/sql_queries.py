"""
SQL query runner and SQL-vs-pandas JOIN verification.

Run after:
    python data_pipeline.py

The five examples collectively demonstrate SELECT, WHERE, ORDER BY,
LIMIT, DISTINCT and JOIN.
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "data" / "books.db"


QUERIES = {
    "1_select": """
        SELECT book_id, title, price_gbp, rating
        FROM books
    """,
    "2_where": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating >= 4
    """,
    "3_order_by_limit": """
        SELECT title, price_inr
        FROM books
        ORDER BY price_inr DESC
        LIMIT 10
    """,
    "4_distinct": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name
    """,
    "5_join": """
        SELECT
            b.title,
            b.rating,
            c.category_name
        FROM books AS b
        JOIN categories AS c
            ON b.category_id = c.category_id
        ORDER BY c.category_name, b.title
    """,
}


def run_queries():
    conn = sqlite3.connect(DB_PATH)

    for name, query in QUERIES.items():
        print(f"\n===== {name} =====")
        result = pd.read_sql_query(query, conn)
        print(result.to_string(index=False))

    # Required SQL JOIN result.
    sql_join = pd.read_sql_query(QUERIES["5_join"], conn)

    # Reproduce the same JOIN directly from in-memory DataFrames.
    books = pd.read_sql_query(
        "SELECT * FROM books", conn
    )
    categories = pd.read_sql_query(
        "SELECT * FROM categories", conn
    )

    pandas_join = (
        books.merge(
            categories,
            on="category_id",
            how="inner",
            suffixes=("_book", "_category"),
        )
        [["title", "rating", "category_name"]]
        .sort_values(["category_name", "title"])
        .reset_index(drop=True)
    )

    sql_join = sql_join.reset_index(drop=True)

    print("\n===== SQL JOIN vs Pandas merge =====")
    print("Equivalent:", sql_join.equals(pandas_join))

    if not sql_join.equals(pandas_join):
        print("\nSQL result:")
        print(sql_join.to_string(index=False))
        print("\nPandas result:")
        print(pandas_join.to_string(index=False))

    conn.close()


if __name__ == "__main__":
    run_queries()
