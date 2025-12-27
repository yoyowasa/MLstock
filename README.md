# MLStock

Reference data ingestion for Alpaca assets and market calendar.

## Setup

1) Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Copy `.env.example` to `.env` and set your Alpaca keys.

3) Review `config.yaml` (override with `config.local.yaml` if needed).

## Run

```bash
python scripts/run_ingest_assets.py
python scripts/run_ingest_calendar.py
python scripts/run_validate_reference.py
```

Outputs:
- `data/reference/assets.parquet`
- `data/reference/calendar.parquet`
- Logs in `artifacts/logs/`
- Validation report in `artifacts/validate/`
# MLstock
