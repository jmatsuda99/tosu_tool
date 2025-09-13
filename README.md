# Energy Viewer v1.8 (deviation bars + totals)
- Single Day: 3-hour mean dashed lines for selected series.
- Single Day: **Deviation bars** anchored at mean (Before loss only), color-coded by sign.
- In-graph **daily totals** (kWh) for blue(+) / red(-). Totals account for current unit:
  - If kW view: sum(deviation[kW>0])*0.5 and sum(|deviation[kW<0]|)*0.5
  - If kWh view: sum(deviation[kWh>0]) and sum(|deviation[kWh<0]|)
- Minimal column rename preprocessing: `(ロス後)/(ロス前)` → `_ロス後/_ロス前`.
