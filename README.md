# masai-project
Masai Capstone Project
# Zepto AI/ML Capstone Project

This repository contains three modules:

1. **Analytics** — data analysis, visualization, and business insights.
2. **Data Pipeline** — data extraction, cleaning, transformation, database storage, and SQL analysis.
3. **Support Assistant** — a RAG-based Zepto customer-support assistant using ChromaDB, LangGraph, FastAPI, Pydantic, and `all-MiniLM-L6-v2`.

## Project Structure

```text
masai-project/
├── analytics/
├── data_pipeline/
│   └── sql_queries.py
├── support_assistant/
│   ├── docs/
│   │   ├── doc_01.txt
│   │   ├── doc_02.txt
│   │   ├── doc_03.txt
│   │   ├── doc_04.txt
│   │   ├── doc_05.txt
│   │   ├── doc_06.txt
│   │   ├── doc_07.txt
│   │   └── doc_08.txt
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── embeddings.py
│   ├── graph.py
│   ├── ingest.py
│   ├── models.py
│   ├── prompts.py
│   └── vectorstore.py
├── main.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Module 1 — Analytics

The Analytics module analyzes the Zepto dataset and produces useful business insights.

Typical workflow:

```text
Raw Data
   ↓
Data Loading
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Business Metrics
   ↓
Visualizations
   ↓
Insights
```

Technologies:
- Python
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook / Python

## Module 2 — Data Pipeline

The Data Pipeline module handles data collection, processing, storage, and SQL analysis.

Typical workflow:

```text
Source Data
    ↓
Data Extraction
    ↓
Data Cleaning
    ↓
Data Transformation
    ↓
SQLite / Database
    ↓
SQL Queries
    ↓
Results
```

Technologies:
- Python
- Requests
- BeautifulSoup
- Pandas
- SQLite
- SQL

## Module 3 — Support Assistant

The Support Assistant is a RAG application for answering Zepto customer-support questions from a fixed policy corpus.

### Eight documents

```text
doc_01.txt — Delivery Policy
doc_02.txt — Returns & Refunds
doc_03.txt — Membership Tiers
doc_04.txt — Order Tracking
doc_05.txt — Order Cancellation Policy
doc_06.txt — Damaged or Missing Items
doc_07.txt — Gift Cards
doc_08.txt — Customer Support Hours
```

### RAG architecture

```text
Policy Documents
       ↓
Document Loading
       ↓
Chunking
       ↓
all-MiniLM-L6-v2
       ↓
Embeddings
       ↓
ChromaDB
       ↓
User Query
       ↓
Intent Classification
       ↓
 ┌─────────────────────┐
 │                     │
Policy Question    General Question
 │                     │
 ↓                     ↓
Retrieve Top 3     Direct Answer
Chunks
 │
 ↓
Answer
 │
 └──────────┬──────────┘
            ↓
      Pydantic JSON
            ↓
         FastAPI
            ↓
         POST /ask
```

### LangGraph

The graph contains:

```text
classify_intent
       │
       ├── policy_question → retrieve_and_answer
       │
       └── general_question → direct_answer
```

Policy-related keywords include:

```text
delivery
return
refund
membership
tracking
track
cancel
gift card
support hours
```

### Embeddings

The application uses:

```text
all-MiniLM-L6-v2
```

The embedding model runs locally and does not require an API key.

### MOCK_LLM

The default graded path is deterministic:

```text
MOCK_LLM=1
```

No external LLM API key is required for the baseline implementation.

## Installation

From the project root:

```powershell
python -m venv venv
.env\Scripts\activate
python -m pip install -r requirements.txt
```

## Run the Data Pipeline

Use the scripts inside:

```text
data_pipeline/
```

For example:

```powershell
python data_pipeline/pipeline.py
```

Use the module's own entry point if it differs.

## Run the Support Assistant

### 1. Index the documents

Keep the eight documents inside:

```text
support_assistant/docs/
```

Then run from the project root:

```powershell
python -m support_assistant.ingest
```

A successful run should report that 8 documents/chunks were indexed.

### 2. Start FastAPI

From the project root:

```powershell
python -m uvicorn support_assistant.api:app --host 127.0.0.1 --port 7860
```

If the project-level `main.py` imports the FastAPI app, this also works:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 7860
```

### 3. Open Swagger

Open:

```text
http://127.0.0.1:7860/docs
```

Select **POST /ask**, click **Try it out**, and enter:

```json
{
  "query": "What is the delivery fee for orders below INR 149?"
}
```

Example response:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": ["doc_01.txt"],
  "confidence": 0.85
}
```

## Example Questions

```text
What is the delivery fee for orders below INR 149?
How much does Zepto Pass cost?
Can I return a damaged grocery item?
How can I track my order?
Can I cancel my order after it is packed?
What gift card denominations are available?
What are Zepto customer support hours?
```

## API

### POST `/ask`

Request:

```json
{
  "query": "How much does Zepto Pass cost?"
}
```

Response:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": ["doc_03.txt"],
  "confidence": 0.85
}
```

Fields:

| Field | Description |
|---|---|
| `answer` | Customer-facing answer |
| `sources` | Documents/chunks used |
| `confidence` | Value from 0 to 1 |

## Docker

Build:

```powershell
docker build -t zepto-support-assistant .
```

Run:

```powershell
docker run -p 7860:7860 zepto-support-assistant
```

Then open:

```text
http://127.0.0.1:7860/docs
```

## Overall Architecture

```text
                    Zepto AI/ML Capstone
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ↓                 ↓                 ↓
      Analytics       Data Pipeline    Support Assistant
          │                 │                 │
          ↓                 ↓                 ↓
       Insights        Processed Data     RAG Answers
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ↓
                     Complete Project
```

Analytics focuses on **understanding data**, Data Pipeline focuses on **collecting and processing data**, and Support Assistant focuses on **retrieving policy information and answering customer questions**.

## Technology Stack

| Area | Technology |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib |
| Web/Data Extraction | Requests, BeautifulSoup |
| Database | SQLite |
| Vector Database | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 |
| Workflow | LangGraph |
| API | FastAPI |
| Validation | Pydantic |
| Server | Uvicorn |
| Containerization | Docker |
| Version Control | Git / GitHub |

## Completion Checklist

- [ ] Analytics module works.
- [ ] Data Pipeline works.
- [ ] SQL queries execute successfully.
- [ ] All 8 Support Assistant documents are present.
- [ ] Documents are embedded with `all-MiniLM-L6-v2`.
- [ ] ChromaDB is populated.
- [ ] LangGraph conditional routing works.
- [ ] `MOCK_LLM` baseline works without an API key.
- [ ] Pydantic response validation works.
- [ ] `POST /ask` works.
- [ ] Swagger `/docs` works.
- [ ] Docker build succeeds.
- [ ] Docker container runs the API.
- [ ] README documents all three modules.

## Important Notes

- Keep the Support Assistant documents in `support_assistant/docs/`.
- Run Python module commands from the `masai-project` root.
- Do not run `graph.py` or `api.py` directly when they use package-relative imports.
- Use `python -m support_assistant.ingest` for ingestion.
- Use Uvicorn to run the FastAPI application.
