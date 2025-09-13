# Energy Viewer v1.8

**What’s new**

- Selectable shaded band (±X, step 100) on *Single Day* and *Daily Overlay* charts.
- CSV downloads added for all charts, plus **Full DB export** (Excel/CSV).
- Keeps English titles & legend toggle (default OFF).
- 30-min kWh → kW conversion toggle is available in every chart.
- Import now does idempotent **upsert** keyed by `開始日時`.

## Files

- `app.py` — Streamlit app entry point (v1.8)
- `requirements.txt` — Python dependencies
