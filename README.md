
# Energy Viewer v1.4 (Time Series + Distribution)

Adds a **Distribution** tab with a histogram overlaid with **Median** and **±1σ/±2σ/±3σ** lines.
- Series selector (after/before loss)
- Unit toggle (kWh / kW): for kW, 30-min kWh are multiplied by 2
- Bins selector
- Stats table and CSV export of the raw vector

Also includes the **Time Series** tab with kWh/kW toggle, daily/monthly aggregation, DB save/load, CSV export, Clear DB.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
