
# Energy Consumption Viewer (Before/After Loss) v1.3

- **English graph labels** (keep original Japanese column names in data)
- **Unit toggle (kWh / kW)**:
  - 30-min view: kW = kWh * 2
  - Daily/Monthly views: shows **Average Power [kW]** (= mean of 30-min kWh * 2) or **Energy [kWh]** (sum)
- Saves uploaded Excel data to **SQLite DB**; subsequent runs load from DB
- Single matplotlib plot; CSV export of displayed data; Clear DB button

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
