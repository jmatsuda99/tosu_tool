
import os
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="Energy Viewer v1.6.1", layout="wide")
st.title("Energy Consumption Viewer — v1.6.1")
st.caption("English titles + legend toggle (default OFF). Data are saved to a local DB.")

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

def to_numeric_safe(s):
    return pd.to_numeric(
        s.astype(str).str.replace(',', '', regex=False).str.strip(),
        errors='coerce'
    )

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
    return df[keep].copy()

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
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} ORDER BY 開始日時", conn)
    df["開始日時"] = pd.to_datetime(df["開始日時"], errors="coerce")
    return df.dropna(subset=["開始日時"])

def clear_db():
    if os.path.exists(DB_PATH):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        os.remove(DB_PATH)

# ---------- UI: file controls ----------
init_db()
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], key="upload_xlsx")
if uploaded is not None:
    df_up = load_excel(uploaded)
    if not df_up.empty:
        save_to_db(df_up)
        st.success(f"Saved to DB. Rows: {len(df_up):,}")

if st.button("Clear DB", key="btn_clear_db"):
    clear_db()
    st.warning("DB cleared. Upload again to repopulate.")

df = load_from_db()
if df.empty:
    default_path = "/mnt/data/240801 24年8月～25年7月鳥栖PO1期.xlsx"
    if os.path.exists(default_path):
        df_demo = load_excel(default_path)
        if not df_demo.empty:
            save_to_db(df_demo)
            df = load_from_db()
if df.empty:
    st.stop()

min_dt, max_dt = df["開始日時"].min(), df["開始日時"].max()
st.write(f"Data range: **{min_dt}** to **{max_dt}**")

# ---------- Tabs ----------
tab_ts, tab_dist, tab_day, tab_overlay = st.tabs(["Time Series", "Distribution", "Daily View", "Daily Overlay"])

# ---------- Time Series ----------
with tab_ts:
    view = st.radio("Granularity", ["30-min (raw)", "Daily (sum/avg kW)", "Monthly (sum/avg kW)"], horizontal=True, key="ts_granularity")
    unit = st.radio("Unit", ["kWh", "kW"], horizontal=True, key="ts_unit")
    show_legend = st.checkbox("Show legend", value=False, key="ts_legend")

    y_cols = ["使用電力量_ロス後","使用電力量_ロス前"]
    plot_df = df[["開始日時"]+y_cols].copy()
    plot_df[y_cols] = plot_df[y_cols].apply(to_numeric_safe)

    if view == "30-min (raw)":
        display_df = plot_df.sort_values("開始日時")
        if unit=="kW":
            display_df[y_cols] *= 2.0
        title = "Energy Consumption (Before/After Loss)" if unit=="kWh" else "Power (Before/After Loss)"
    elif view.startswith("Daily"):
        grouped = plot_df.set_index("開始日時").resample("D")
        if unit=="kW":
            display_df = grouped.mean(numeric_only=True).reset_index()
            display_df[y_cols] *= 2.0
            title="Average Power (Daily)"
        else:
            display_df = grouped.sum(numeric_only=True).reset_index()
            title="Energy (Daily Sum)"
    else:
        grouped = plot_df.set_index("開始日時").resample("MS")
        if unit=="kW":
            display_df = grouped.mean(numeric_only=True).reset_index()
            display_df[y_cols] *= 2.0
            title="Average Power (Monthly)"
        else:
            display_df = grouped.sum(numeric_only=True).reset_index()
            title="Energy (Monthly Sum)"

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(display_df["開始日時"], display_df["使用電力量_ロス後"], label="After loss")
    ax.plot(display_df["開始日時"], display_df["使用電力量_ロス前"], label="Before loss")
    ax.set_title(title)
    ax.set_xlabel("Date/Time")
    ax.set_ylabel("Power [kW]" if unit=="kW" else "Energy [kWh]")
    ax.grid(True)
    if show_legend: ax.legend()
    st.pyplot(fig, clear_figure=True)

# ---------- Distribution ----------
with tab_dist:
    target = st.selectbox("Target", ["使用電力量_ロス後","使用電力量_ロス前"], key="dist_target")
    unit2 = st.radio("Unit", ["kWh","kW"], horizontal=True, key="dist_unit")
    bins = st.number_input("Bins", min_value=10, max_value=200, value=50, step=5, key="dist_bins")
    show_legend = st.checkbox("Show legend", value=False, key="dist_legend")

    vec = to_numeric_safe(df[target]).dropna()
    if unit2=="kW": vec*=2.0

    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.hist(vec, bins=int(bins))
    ax2.set_title("Histogram with Median and ±nσ")
    ax2.set_xlabel("Power [kW]" if unit2=="kW" else "Energy [kWh]")
    ax2.set_ylabel("Count")

    median, sigma = float(np.median(vec)), float(np.std(vec, ddof=0))
    ax2.axvline(median, linestyle="-", label="Median")
    for n,style in zip([1,2,3],["--","-.",":"]):
        ax2.axvline(median-n*sigma, linestyle=style, label=f"-{n}σ")
        ax2.axvline(median+n*sigma, linestyle=style, label=f"+{n}σ")
    ax2.grid(True)
    if show_legend: ax2.legend()
    st.pyplot(fig2, clear_figure=True)

# ---------- Daily View ----------
with tab_day:
    unit3 = st.radio("Unit",["kWh","kW"],horizontal=True,key="day_unit")
    series_choice = st.multiselect("Series",["使用電力量_ロス後","使用電力量_ロス前"],["使用電力量_ロス後","使用電力量_ロス前"], key="day_series")
    day = st.date_input("Pick a date", key="day_date")
    show_legend = st.checkbox("Show legend", value=False, key="day_legend")

    if day:
        start_dt = datetime.combine(day, datetime.min.time())
        end_dt = start_dt+timedelta(days=1)
        df_day = df[(df["開始日時"]>=start_dt)&(df["開始日時"]<end_dt)].copy()
        if not df_day.empty:
            for c in series_choice:
                df_day[c] = to_numeric_safe(df_day[c])
                if unit3=="kW": df_day[c]*=2.0
            df_day["time"]=df_day["開始日時"].dt.strftime("%H:%M")
            fig3, ax3 = plt.subplots(figsize=(10,4))
            for c in series_choice:
                ax3.plot(df_day["time"], df_day[c], label=("After loss" if c.endswith("後") else "Before loss"))
            ax3.set_title(f"Single Day — {day.isoformat()}")
            ax3.set_xlabel("Time of Day"); ax3.set_ylabel("Power [kW]" if unit3=="kW" else "Energy [kWh]")
            ax3.grid(True)
            if show_legend: ax3.legend()
            st.pyplot(fig3, clear_figure=True)

# ---------- Daily Overlay ----------
with tab_overlay:
    unit4 = st.radio("Unit",["kWh","kW"],horizontal=True,key="overlay_unit")
    target = st.selectbox("Target",["使用電力量_ロス後","使用電力量_ロス前"], key="overlay_target")
    start_date = st.date_input("Start date", key="overlay_start")
    end_date = st.date_input("End date", key="overlay_end")
    show_legend = st.checkbox("Show legend", value=False, key="overlay_legend")

    if start_date and end_date and start_date<=end_date:
        fig4, ax4 = plt.subplots(figsize=(10,5))
        for i in range((end_date-start_date).days+1):
            d = start_date+timedelta(days=i)
            start_dt=datetime.combine(d,datetime.min.time())
            end_dt=start_dt+timedelta(days=1)
            df_i=df[(df["開始日時"]>=start_dt)&(df["開始日時"]<end_dt)].copy()
            if df_i.empty: continue
            v=to_numeric_safe(df_i[target])
            if unit4=="kW": v*=2.0
            df_i["time"]=df_i["開始日時"].dt.strftime("%H:%M")
            ax4.plot(df_i["time"], v, label=d.isoformat())
        ax4.set_title(f"Daily Overlay — {start_date.isoformat()} to {end_date.isoformat()} ({'After loss' if target.endswith('後') else 'Before loss'})")
        ax4.set_xlabel("Time of Day"); ax4.set_ylabel("Power [kW]" if unit4=="kW" else "Energy [kWh]")
        ax4.grid(True)
        if show_legend: ax4.legend(ncol=2,fontsize="small")
        st.pyplot(fig4, clear_figure=True)
