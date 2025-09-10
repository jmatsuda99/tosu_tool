
import os
import sqlite3
import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Use generic sans-serif fonts (English labels)
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="Energy Viewer (DB, EN Graph)", layout="wide")
st.title("Energy Consumption Viewer (Before/After Loss)")
st.caption("X-axis: Start Time. Y-axis: Energy/Power. Lines: Before vs After loss. Data are saved to a local DB so you don't need to re-upload every time.")

DB_PATH = "data.db"
TABLE_NAME = "usage_data"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME}(
                開始日時 TEXT PRIMARY KEY,
                使用電力量_ロス後 REAL,
                使用電力量_ロス前 REAL
            )
        """)
    return True

@st.cache_data
def load_excel(file):
    df = pd.read_excel(file)
    expected_cols = ["開始日時", "使用電力量(ロス後)", "使用電力量(ロス前)"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.warning(f"Missing columns: {missing}")
    # Normalize column names for DB
    df = df.rename(columns={"使用電力量(ロス後)": "使用電力量_ロス後", "使用電力量(ロス前)": "使用電力量_ロス前"})
    # Parse datetime
    if "開始日時" in df.columns:
        df["開始日時"] = pd.to_datetime(df["開始日時"], errors="coerce")
        df = df.dropna(subset=["開始日時"])
    keep = ["開始日時", "使用電力量_ロス後", "使用電力量_ロス前"]
    df = df[[c for c in keep if c in df.columns]].copy()
    return df

def save_to_db(df):
    df2 = df.copy()
    # Ensure naive datetime strings
    df2["開始日時"] = pd.to_datetime(df2["開始日時"], errors="coerce")
    df2 = df2.dropna(subset=["開始日時"])
    df2["開始日時"] = df2["開始日時"].dt.strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("BEGIN")
        conn.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_NAME}(開始日時 TEXT PRIMARY KEY, 使用電力量_ロス後 REAL, 使用電力量_ロス前 REAL)")
        for _, row in df2.iterrows():
            conn.execute(
                f"INSERT INTO {TABLE_NAME}(開始日時, 使用電力量_ロス後, 使用電力量_ロス前) VALUES (?, ?, ?) "
                f"ON CONFLICT(開始日時) DO UPDATE SET 使用電力量_ロス後=excluded.使用電力量_ロス後, 使用電力量_ロス前=excluded.使用電力量_ロス前",
                (row["開始日時"], float(row.get("使用電力量_ロス後") or 0), float(row.get("使用電力量_ロス前") or 0))
            )
        conn.commit()

@st.cache_data
def load_from_db():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=["開始日時","使用電力量_ロス後","使用電力量_ロス前"])
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(f"SELECT 開始日時, 使用電力量_ロス後, 使用電力量_ロス前 FROM {TABLE_NAME} ORDER BY 開始日時", conn)
    if not df.empty:
        df["開始日時"] = pd.to_datetime(df["開始日時"], errors="coerce")
        df = df.dropna(subset=["開始日時"])
    return df

def clear_db():
    if os.path.exists(DB_PATH):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        os.remove(DB_PATH)

# UI — Data input
init_db()
left, right = st.columns([2,1], gap="large")
with left:
    uploaded = st.file_uploader("Upload Excel (.xlsx) — only first time. Data will be saved to DB.", type=["xlsx"])
    if uploaded is not None:
        df_up = load_excel(uploaded)
        if not df_up.empty:
            save_to_db(df_up)
            st.success(f"Saved to DB. Rows: {len(df_up):,}")
        else:
            st.warning("No valid rows found.")

with right:
    if st.button("Clear DB", type="secondary"):
        clear_db()
        st.warning("DB cleared. Reload to apply.")

# Load data from DB (bootstrap from default path if empty)
df = load_from_db()
if df.empty:
    default_path = "/mnt/data/240801 24年8月～25年7月鳥栖PO1期.xlsx"
    if os.path.exists(default_path):
        st.info("DB is empty. Loaded a default file and saved to DB.")
        df_demo = load_excel(default_path)
        if not df_demo.empty:
            save_to_db(df_demo)
            df = load_from_db()
    else:
        st.stop()

# Controls
view = st.radio("Granularity", ["30-min (raw)", "Daily (sum / avg kW)", "Monthly (sum / avg kW)"], horizontal=True)
unit = st.radio("Unit", ["kWh", "kW"], horizontal=True)

plot_df = df.copy()
y_cols = ["使用電力量_ロス後", "使用電力量_ロス前"]
plot_df = plot_df[["開始日時"] + [c for c in y_cols if c in plot_df.columns]].copy()

# Aggregate
if view == "30-min (raw)":
    display_df = plot_df.sort_values("開始日時").copy()
    if unit == "kW":
        for c in y_cols:
            if c in display_df.columns:
                display_df[c] = display_df[c] * 2.0  # 30-min kWh -> kW
    x_col = "開始日時"
    y_label = "Power [kW]" if unit == "kW" else "Energy [kWh]"
    title = "Energy/Power (30-min)"
elif view == "Daily (sum / avg kW)":
    # Sum for kWh; Avg power for kW (mean of 30-min kWh * 2)
    grouped = (plot_df.set_index("開始日時").resample("D"))
    if unit == "kW":
        display_df = grouped.mean(numeric_only=True).reset_index()
        for c in y_cols:
            if c in display_df.columns:
                display_df[c] = display_df[c] * 2.0
        x_col = "開始日時"
        y_label = "Average Power [kW]"
        title = "Average Power (Daily)"
    else:
        display_df = grouped.sum(numeric_only=True).reset_index()
        x_col = "開始日時"
        y_label = "Energy [kWh]"
        title = "Energy (Daily Sum)"
elif view == "Monthly (sum / avg kW)":
    grouped = (plot_df.set_index("開始日時").resample("MS"))
    if unit == "kW":
        display_df = grouped.mean(numeric_only=True).reset_index()
        for c in y_cols:
            if c in display_df.columns:
                display_df[c] = display_df[c] * 2.0
        x_col = "開始日時"
        y_label = "Average Power [kW]"
        title = "Average Power (Monthly)"
    else:
        display_df = grouped.sum(numeric_only=True).reset_index()
        x_col = "開始日時"
        y_label = "Energy [kWh]"
        title = "Energy (Monthly Sum)"

# Plot
fig, ax = plt.subplots(figsize=(10, 4))
legend_map = {
    "使用電力量_ロス後": "Energy (after loss)" if unit == "kWh" else "Power (after loss)",
    "使用電力量_ロス前": "Energy (before loss)" if unit == "kWh" else "Power (before loss)",
}
for col in [c for c in y_cols if c in display_df.columns]:
    ax.plot(display_df[x_col], display_df[col], label=legend_map.get(col, col))
ax.set_xlabel("Start Time" if x_col == "開始日時" else ("Date" if "D" in view else "Month"))
ax.set_ylabel(y_label)
ax.set_title("Energy Consumption (Before/After Loss)" if unit == "kWh" else "Power (Before/After Loss)")
ax.grid(True)
ax.legend()
st.pyplot(fig, clear_figure=True)

# Table + CSV
st.subheader("Preview (top 50 rows)")
st.dataframe(display_df.head(50))
csv = display_df.rename(columns={
    "開始日時": "Start Time"
}).to_csv(index=False).encode("utf-8-sig")
st.download_button("Download displayed data (CSV)", data=csv, file_name="display_data.csv", mime="text/csv")
