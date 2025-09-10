
import os
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta

matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="Energy Viewer (TS + Distribution + Daily Compare)", layout="wide")
st.title("Energy Consumption Viewer (Before/After Loss) — v1.5")
st.caption("Now includes Daily View and Daily Overlay by time-of-day. Data are saved to a local DB.")

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

# Top controls
c1, c2, c3 = st.columns([1,1,1])
with c1:
    uploaded = st.file_uploader("Upload Excel (.xlsx) — first time only", type=["xlsx"], key="upload_xlsx")
    if uploaded is not None:
        df_up = load_excel(uploaded)
        if df_up is not None and not df_up.empty:
            save_to_db(df_up)
            st.success(f"Saved to DB. Rows: {len(df_up):,}")
        else:
            st.warning("No valid rows found in the uploaded file.")
with c2:
    if st.button("Clear DB", key="btn_clear_db"):
        clear_db()
        st.warning("DB cleared. Upload again or rely on default file.")
with c3:
    if st.button("Clear Cache", key="btn_clear_cache"):
        st.cache_data.clear()
        st.success("Cache cleared. Reload the page.")

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

# Shared info
min_dt, max_dt = df["開始日時"].min(), df["開始日時"].max()
st.write(f"Data range: **{min_dt}** to **{max_dt}**")

# Tabs
tab_ts, tab_dist, tab_day, tab_overlay = st.tabs(["Time Series", "Distribution", "Daily View", "Daily Overlay"])

# ==================== Time Series Tab ====================
with tab_ts:
    view = st.radio("Granularity", ["30-min (raw)", "Daily (sum / avg kW)", "Monthly (sum / avg kW)"], horizontal=True, key="ts_granularity")
    unit = st.radio("Unit", ["kWh", "kW"], horizontal=True, key="ts_unit")

    y_cols = ["使用電力量_ロス後", "使用電力量_ロス前"]
    plot_df = df[["開始日時"] + y_cols].copy()
    plot_df[y_cols] = plot_df[y_cols].apply(to_numeric_safe)

    if plot_df[y_cols].dropna(how="all").empty:
        st.warning("Both series are fully NaN after numeric conversion.")
    else:
        if view == "30-min (raw)":
            display_df = plot_df.sort_values("開始日時").copy()
            if unit == "kW":
                for c in y_cols:
                    display_df[c] = display_df[c] * 2.0
            x_col = "開始日時"
            y_label = "Power [kW]" if unit == "kW" else "Energy [kWh]"
        elif view == "Daily (sum / avg kW)":
            grouped = (plot_df.set_index("開始日時").sort_index().resample("D"))
            if unit == "kW":
                display_df = grouped.mean(numeric_only=True).reset_index()
                for c in y_cols:
                    display_df[c] = display_df[c] * 2.0
                x_col = "開始日時"; y_label = "Average Power [kW]"
            else:
                display_df = grouped.sum(numeric_only=True).reset_index()
                x_col = "開始日時"; y_label = "Energy [kWh]"
        elif view == "Monthly (sum / avg kW)":
            grouped = (plot_df.set_index("開始日時").sort_index().resample("MS"))
            if unit == "kW":
                display_df = grouped.mean(numeric_only=True).reset_index()
                for c in y_cols:
                    display_df[c] = display_df[c] * 2.0
                x_col = "開始日時"; y_label = "Average Power [kW]"
            else:
                display_df = grouped.sum(numeric_only=True).reset_index()
                x_col = "開始日時"; y_label = "Energy [kWh]"

        display_df = display_df.dropna(subset=[x_col], how="any")
        if display_df.empty or (display_df[y_cols].isna().all().all()):
            st.warning("Nothing to plot: aggregated series are empty or all NaN.")
        else:
            display_df = display_df.sort_values(x_col)
            fig, ax = plt.subplots(figsize=(10, 4))
            legend_map = {
                "使用電力量_ロス後": "Energy (after loss)" if unit == "kWh" else "Power (after loss)",
                "使用電力量_ロス前": "Energy (before loss)" if unit == "kWh" else "Power (before loss)",
            }
            for col in y_cols:
                if col in display_df.columns and display_df[col].notna().any():
                    ax.plot(display_df[x_col], display_df[col], label=legend_map.get(col, col))
            ax.set_xlabel("Start Time" if x_col == "開始日時" else ("Date" if view.startswith("Daily") else "Month"))
            ax.set_ylabel(y_label)
            ax.set_title("Energy Consumption (Before/After Loss)" if unit == "kWh" else "Power (Before/After Loss)")
            ax.grid(True)
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

    vec = to_numeric_safe(df[target_label]).dropna()
    if unit2 == "kW":
        vec = vec * 2.0

    if vec.empty:
        st.warning("Selected series has no numeric data after conversion.")
    else:
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

# ==================== Daily View (new) ====================
with tab_day:
    unit3 = st.radio("Unit", ["kWh", "kW"], horizontal=True, key="day_unit")
    series_choice = st.multiselect("Series", ["使用電力量_ロス後", "使用電力量_ロス前"], default=["使用電力量_ロス後","使用電力量_ロス前"], key="day_series")
    day = st.date_input("Pick a date (single day)", key="day_date")
    if isinstance(day, list) or isinstance(day, tuple):
        if len(day) > 0:
            day = day[0]
        else:
            day = None
    if day is None:
        st.info("Select a date.")
    else:
        start_dt = datetime.combine(day, datetime.min.time())
        end_dt = start_dt + timedelta(days=1)
        df_day = df[(df["開始日時"] >= start_dt) & (df["開始日時"] < end_dt)].copy()
        if df_day.empty:
            st.warning("No records for the selected date.")
        else:
            df_day = df_day.sort_values("開始日時")
            for c in series_choice:
                df_day[c] = to_numeric_safe(df_day[c])
                if unit3 == "kW":
                    df_day[c] = df_day[c] * 2.0
            df_day["time_of_day"] = df_day["開始日時"].dt.strftime("%H:%M")
            fig3, ax3 = plt.subplots(figsize=(10,4))
            for c in series_choice:
                if c in df_day.columns and df_day[c].notna().any():
                    ax3.plot(df_day["time_of_day"], df_day[c], label=("After loss" if c=="使用電力量_ロス後" else "Before loss"))
            ax3.set_xlabel("Time of Day")
            ax3.set_ylabel("Power [kW]" if unit3=="kW" else "Energy [kWh]")
            ax3.set_title(f"Single Day — {day.isoformat()}")
            ax3.grid(True)
            ax3.legend()
            st.pyplot(fig3, clear_figure=True)
            st.subheader("Preview (top 50 rows)")
            st.dataframe(df_day[["開始日時"] + series_choice].head(50))
            out = df_day[["開始日時","time_of_day"] + series_choice].rename(columns={"開始日時":"Start Time"})
            st.download_button("Download single-day data (CSV)", data=out.to_csv(index=False).encode("utf-8-sig"), file_name="single_day.csv", mime="text/csv", key="dl_day_csv")

# ==================== Daily Overlay (new) ====================
with tab_overlay:
    unit4 = st.radio("Unit", ["kWh", "kW"], horizontal=True, key="overlay_unit")
    target = st.selectbox("Target series", ["使用電力量_ロス後", "使用電力量_ロス前"], index=0, key="overlay_target")
    dcol1, dcol2 = st.columns(2)
    start_date = dcol1.date_input("Start date", key="overlay_start")
    end_date = dcol2.date_input("End date", key="overlay_end")
    max_days = 60
    if start_date and end_date:
        if start_date > end_date:
            st.error("Start date must be on or before End date.")
        else:
            days = (end_date - start_date).days + 1
            if days > max_days:
                st.warning(f"Range is {days} days; limiting to the first {max_days} days to keep the chart readable.")
                end_date = start_date + timedelta(days=max_days-1)
            fig4, ax4 = plt.subplots(figsize=(10,5))
            overlay_rows = []
            for i in range((end_date - start_date).days + 1):
                day_i = start_date + timedelta(days=i)
                start_dt = datetime.combine(day_i, datetime.min.time())
                end_dt = start_dt + timedelta(days=1)
                df_i = df[(df['開始日時'] >= start_dt) & (df['開始日時'] < end_dt)].copy()
                if df_i.empty:
                    continue
                vec = to_numeric_safe(df_i[target])
                if unit4 == "kW":
                    vec = vec * 2.0
                df_i = df_i.assign(value=vec)
                df_i = df_i.dropna(subset=['value'])
                if df_i.empty:
                    continue
                df_i = df_i.sort_values('開始日時')
                df_i['time_of_day'] = df_i['開始日時'].dt.strftime('%H:%M')
                label = f"{day_i.isoformat()}"
                ax4.plot(df_i['time_of_day'], df_i['value'], label=label)
                overlay_rows.append(df_i.assign(day=label)[['day','開始日時','time_of_day','value']])
            if not overlay_rows:
                st.warning("No data found for the specified range.")
            else:
                ax4.set_xlabel("Time of Day")
                ax4.set_ylabel("Power [kW]" if unit4=="kW" else "Energy [kWh]")
                ax4.set_title(f"Daily Overlay — {start_date.isoformat()} to {end_date.isoformat()} ({target})")
                ax4.grid(True)
                ax4.legend(ncol=2, fontsize='small')
                st.pyplot(fig4, clear_figure=True)
                cat = pd.concat(overlay_rows, ignore_index=True)
                cat = cat.rename(columns={"開始日時":"Start Time", "value": ("Power [kW]" if unit4=="kW" else "Energy [kWh]")})
                st.download_button("Download overlay data (CSV)", data=cat.to_csv(index=False).encode("utf-8-sig"), file_name="daily_overlay.csv", mime="text/csv", key="dl_overlay_csv")
    else:
        st.info("Pick both Start date and End date.")
