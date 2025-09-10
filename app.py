
import os
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="Energy Viewer (Time Series + Distribution)", layout="wide")
st.title("Energy Consumption Viewer (Before/After Loss)")
st.caption("Time series and distribution views. Data are saved to a local DB so you don't need to re-upload every time.")

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

@st.cache_data
def load_excel(file):
    df = pd.read_excel(file)
    col_map = {"使用電力量(ロス後)": "使用電力量_ロス後", "使用電力量(ロス前)": "使用電力量_ロス前"}
    df = df.rename(columns=col_map)
    if "開始日時" in df.columns:
        df["開始日時"] = pd.to_datetime(df["開始日時"], errors="coerce")
    df = df.dropna(subset=["開始日時"])
    keep = ["開始日時", "使用電力量_ロス後", "使用電力量_ロス前"]
    for k in keep:
        if k not in df.columns:
            df[k] = pd.NA
    df = df[keep].copy()
    return df

def save_to_db(df):
    df2 = df.copy()
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
                (row["開始日時"],
                 None if pd.isna(row["使用電力量_ロス後"]) else float(row["使用電力量_ロス後"]),
                 None if pd.isna(row["使用電力量_ロス前"]) else float(row["使用電力量_ロス前"]))
            )
        conn.commit()

@st.cache_data
def load_from_db():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=["開始日時","使用電力量_ロス後","使用電力量_ロス前"])
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(f"SELECT 開始日時, 使用電力量_ロス後, 使用電力量_ロス前 FROM {TABLE_NAME} ORDER BY 開始日時", conn)
    if df.empty:
        return df
    df["開始日時"] = pd.to_datetime(df["開始日時"], errors="coerce")
    df = df.dropna(subset=["開始日時"])
    return df

def clear_db():
    if os.path.exists(DB_PATH):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        os.remove(DB_PATH)

init_db()

# Upload / DB controls
left, right = st.columns([2,1], gap="large")
with left:
    uploaded = st.file_uploader("Upload Excel (.xlsx) — only first time. Data will be saved to DB.", type=["xlsx"], key="upload_xlsx")
    if uploaded is not None:
        df_up = load_excel(uploaded)
        if df_up is not None and not df_up.empty:
            save_to_db(df_up)
            st.success(f"Saved to DB. Rows: {len(df_up):,}")
        else:
            st.warning("No valid rows found in the uploaded file.")

with right:
    if st.button("Clear DB", type="secondary", key="btn_clear_db"):
        clear_db()
        st.warning("DB cleared. Reload to apply.")

# Load data (or bootstrap)
df = load_from_db()
if df.empty:
    default_path = "/mnt/data/240801 24年8月～25年7月鳥栖PO1期.xlsx"
    if os.path.exists(default_path):
        st.info("DB is empty. Loaded a default file and saved to DB.")
        df_demo = load_excel(default_path)
        if df_demo is not None and not df_demo.empty:
            save_to_db(df_demo)
            df = load_from_db()
    if df.empty:
        st.stop()

# Shared controls
min_dt, max_dt = df["開始日時"].min(), df["開始日時"].max()
st.write(f"Data range: **{min_dt}** to **{max_dt}**")
range_sel = st.slider("Filter by date range", min_value=min_dt.to_pydatetime(), max_value=max_dt.to_pydatetime(), value=(min_dt.to_pydatetime(), max_dt.to_pydatetime()), key="date_range_slider")
mask = (df["開始日時"] >= pd.to_datetime(range_sel[0])) & (df["開始日時"] <= pd.to_datetime(range_sel[1]))
df = df.loc[mask].copy()
if df.empty:
    st.warning("No data in the selected date range.")
    st.stop()

tab_ts, tab_dist = st.tabs(["Time Series", "Distribution"])

# ==================== Time Series Tab ====================
with tab_ts:
    view = st.radio("Granularity", ["30-min (raw)", "Daily (sum / avg kW)", "Monthly (sum / avg kW)"], horizontal=True, key="ts_granularity")
    unit = st.radio("Unit", ["kWh", "kW"], horizontal=True, key="ts_unit")

    y_cols = ["使用電力量_ロス後", "使用電力量_ロス前"]
    for c in y_cols:
        if c not in df.columns:
            st.error(f"Column not found: {c}")
            st.stop()

    plot_df = df[["開始日時"] + y_cols].copy()
    plot_df[y_cols] = plot_df[y_cols].apply(pd.to_numeric, errors="coerce")

    if view == "30-min (raw)":
        display_df = plot_df.sort_values("開始日時").copy()
        if unit == "kW":
            for c in y_cols:
                display_df[c] = display_df[c] * 2.0  # 30-min kWh -> kW
        x_col = "開始日時"
        y_label = "Power [kW]" if unit == "kW" else "Energy [kWh]"
        title = "Energy/Power (30-min)"
    elif view == "Daily (sum / avg kW)":
        grouped = (plot_df.set_index("開始日時").resample("D"))
        if unit == "kW":
            display_df = grouped.mean(numeric_only=True).reset_index()
            for c in y_cols:
                display_df[c] = display_df[c] * 2.0
            x_col = "開始日時"; y_label = "Average Power [kW]"; title = "Average Power (Daily)"
        else:
            display_df = grouped.sum(numeric_only=True).reset_index()
            x_col = "開始日時"; y_label = "Energy [kWh]"; title = "Energy (Daily Sum)"
    elif view == "Monthly (sum / avg kW)":
        grouped = (plot_df.set_index("開始日時").resample("MS"))
        if unit == "kW":
            display_df = grouped.mean(numeric_only=True).reset_index()
            for c in y_cols:
                display_df[c] = display_df[c] * 2.0
            x_col = "開始日時"; y_label = "Average Power [kW]"; title = "Average Power (Monthly)"
        else:
            display_df = grouped.sum(numeric_only=True).reset_index()
            x_col = "開始日時"; y_label = "Energy [kWh]"; title = "Energy (Monthly Sum)"

    display_df = display_df.dropna(subset=[x_col], how="any")
    if display_df.empty or display_df[y_cols].dropna(how="all").empty:
        st.warning("Nothing to plot: data after aggregation is empty.")
    else:
        display_df = display_df.sort_values(x_col)
        fig, ax = plt.subplots(figsize=(10, 4))
        legend_map = {
            "使用電力量_ロス後": "Energy (after loss)" if unit == "kWh" else "Power (after loss)",
            "使用電力量_ロス前": "Energy (before loss)" if unit == "kWh" else "Power (before loss)",
        }
        plotted = False
        for col in y_cols:
            if col in display_df.columns and display_df[col].notna().any():
                ax.plot(display_df[x_col], display_df[col], label=legend_map.get(col, col))
                plotted = True
        ax.set_xlabel("Start Time" if x_col == "開始日時" else ("Date" if "Daily" in view else "Month"))
        ax.set_ylabel(y_label)
        ax.set_title("Energy Consumption (Before/After Loss)" if unit == "kWh" else "Power (Before/After Loss)")
        ax.grid(True)
        if plotted:
            ax.legend()
        st.pyplot(fig, clear_figure=True)

    st.subheader("Preview (top 50 rows)")
    st.dataframe(display_df.head(50))
    csv = display_df.rename(columns={"開始日時": "Start Time"}).to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download displayed data (CSV)", data=csv, file_name="display_data_timeseries.csv", mime="text/csv", key="dl_timeseries_csv")

# ==================== Distribution Tab ====================
with tab_dist:
    col_sel, col_unit, col_bins = st.columns([2,1,1])
    target_label = col_sel.selectbox("Target series", ["使用電力量_ロス後", "使用電力量_ロス前"], index=0, key="dist_target")
    unit2 = col_unit.radio("Unit", ["kWh", "kW"], horizontal=True, key="dist_unit")
    bins = col_bins.number_input("Bins", min_value=10, max_value=200, value=50, step=5, key="dist_bins")

    vec = pd.to_numeric(df[target_label], errors="coerce").dropna()
    if vec.empty:
        st.warning("Selected series has no numeric data.")
        st.stop()

    if unit2 == "kW":
        vec = vec * 2.0

    median = float(np.median(vec))
    sigma = float(np.std(vec, ddof=0))
    stats_df = pd.DataFrame({
        "Metric": ["Median", "-1σ", "+1σ", "-2σ", "+2σ", "-3σ", "+3σ"],
        "Value": [median, median - sigma, median + sigma, median - 2*sigma, median + 2*sigma, median - 3*sigma, median + 3*sigma]
    })

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.hist(vec, bins=int(bins))
    ax2.set_xlabel("Power [kW]" if unit2 == "kW" else "Energy [kWh]")
    ax2.set_ylabel("Count")
    ax2.set_title("Histogram with Median and ±nσ")

    ax2.axvline(median, linestyle='-', linewidth=2, label=f"Median = {median:.3f}")
    ax2.axvline(median - sigma, linestyle='--', label=f"-1σ = {median - sigma:.3f}")
    ax2.axvline(median + sigma, linestyle='--', label=f"+1σ = {median + sigma:.3f}")
    ax2.axvline(median - 2*sigma, linestyle='-.', label=f"-2σ = {median - 2*sigma:.3f}")
    ax2.axvline(median + 2*sigma, linestyle='-.', label=f"+2σ = {median + 2*sigma:.3f}")
    ax2.axvline(median - 3*sigma, linestyle=':', label=f"-3σ = {median - 3*sigma:.3f}")
    ax2.axvline(median + 3*sigma, linestyle=':', label=f"+3σ = {median + 3*sigma:.3f}")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)

    st.subheader("Statistics")
    st.table(stats_df)

    st.download_button(
        "Download raw vector (CSV)",
        data=pd.DataFrame({("Power [kW]" if unit2=="kW" else "Energy [kWh]"): vec}).to_csv(index=False).encode("utf-8-sig"),
        file_name="distribution_vector.csv",
        mime="text/csv",
        key="dl_distribution_vector"
    )
