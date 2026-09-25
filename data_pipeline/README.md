# Data Pipeline — Starter

## Goal

Build an end-to-end pipeline:

**Books to Scrape → scrape → clean → GBP/INR conversion → SQLite → SQL → pandas verification**

## Fixed conversion rate

The assignment specifies:

**1 GBP = 105.50 INR**

This is a fixed project baseline, not a live exchange rate.

## Setup

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python data_pipeline.py
```

Then:

```bash
python sql_queries.py
```

## Expected outputs

The pipeline should produce at least:

- 60 book rows
- 3 or more categories
- `price_gbp` as numeric
- `price_inr` calculated using 105.50
- `rating` as integer 1–5
- `in_stock` as a Boolean-like SQLite value
- a normalized SQLite database with `categories` and `books`
- at least five SQL queries
- SQL JOIN output compared with a pandas merge

## Cleaning decision

This starter uses median imputation for missing numeric values (`price_gbp`
and `rating`). The decision should be understood and explained in the final
submission.


