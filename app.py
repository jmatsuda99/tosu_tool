
import os
import io
import sqlite3
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ---- Matplotlib basic settings ----
matplotlib.rcParams["font.family"] = ["sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

APP_TITLE = "Energy Consumption Viewer — v1.8"
DB_PATH = "energy_cache.db"
TABLE_NAME = "energy_records"

st.set_page_config(page_title="Energy Viewer v1.8", layout="wide")
st.title(APP_TITLE)

# ---- Utilities ----
def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

@st.cache_data
def load_from_db() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=["開始日時","使用電力量_ロス後","使用電力量_ロス前"])
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} ORDER BY 開始日時", conn)
    if df.empty:
        return df
    df["開始日時"] = pd.to_datetime(df["開始日時"], errors="coerce")
    return df.dropna(subset=["開始日時"])

def save_to_db(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE_NAME}(
                開始日時 TEXT PRIMARY KEY,
                使用電力量_ロス後 REAL,
                使用電力量_ロス前 REAL
            )"""
        )
        df2 = df.copy()
        df2["開始日時"] = pd.to_datetime(df2["開始日時"], errors="coerce")
        df2 = df2.dropna(subset=["開始日時"])
        df2["開始日時"] = df2["開始日時"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df2.to_sql(TABLE_NAME, conn, if_exists="append", index=False)

def upsert_into_db(df: pd.DataFrame) -> None:
    # Upsert by replacing duplicates on 開始日時
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE_NAME}(
                開始日時 TEXT PRIMARY KEY,
                使用電力量_ロス後 REAL,
                使用電力量_ロス前 REAL
            )"""
        )
        df2 = df.copy()
        df2["開始日時"] = pd.to_datetime(df2["開始日時"], errors="coerce")
        df2 = df2.dropna(subset=["開始日時"])
        df2["開始日時"] = df2["開始日時"].dt.strftime("%Y-%m-%d %H:%M:%S")
        # temp table + replace
        df2.to_sql("_tmp_import", conn, if_exists="replace", index=False)
        conn.executescript(f"""
            INSERT OR REPLACE INTO {TABLE_NAME}(開始日時, 使用電力量_ロス後, 使用電力量_ロス前)
            SELECT 開始日時, 使用電力量_ロス後, 使用電力量_ロス前 FROM _tmp_import;
            DROP TABLE _tmp_import;
        """)

def to_excel_bytes(sheets: dict) -> bytes:
    buff = io.BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, index=False, sheet_name=str(name)[:31])
    buff.seek(0)
    return buff.getvalue()

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

def convert_unit(df: pd.DataFrame, cols, unit: str) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = to_numeric_safe(out[c])
        if unit == "kW":
            out[c] = out[c] * 2.0   # 30分kWh → kWへ換算
    return out

# ---- Sidebar: Upload & DB tools ----
st.sidebar.header("Data")
file = st.sidebar.file_uploader("Upload CSV/Excel (30-min data)", type=["csv","xlsx","xls"])
if st.sidebar.button("Clear DB"):
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    st.sidebar.success("Database cleared.")
    st.cache_data.clear()


if file is not None:
    try:
        if file.name.lower().endswith(".csv"):
            src = pd.read_csv(file)
        else:
            src = pd.read_excel(file)
        # === 最小前処理：列名変換＆必要列のみ抽出（機能拡張なし） ===
        src.columns = [str(c).strip() for c in src.columns]
        rename_map = {
            "使用電力量(ロス後)": "使用電力量_ロス後",
            "使用電力量(ロス前)": "使用電力量_ロス前",
        }
        src = src.rename(columns=rename_map)
        keep = [c for c in ["開始日時","使用電力量_ロス後","使用電力量_ロス前"] if c in src.columns]
        if keep:
            src = src[keep]
        # === ここまで ===
        upsert_into_db(src)
        st.sidebar.success(f"Imported {len(src)} rows.")
        st.cache_data.clear()
    except Exception as e:
        st.sidebar.error(f"Import failed: {e}")
df = load_from_db()
st.caption(f"Records in DB: {len(df)}")

# ---- Section 1: Time Series (full) ----
st.subheader("Time Series")
if df.empty:
    st.info("Upload data to view charts.")
else:
    unit1 = st.radio("Unit", ["kWh","kW"], horizontal=True, key="ts_unit")
    series_choice = st.multiselect(
        "Series", ["使用電力量_ロス後","使用電力量_ロス前"], default=["使用電力量_ロス後"], key="ts_series"
    )
    show_legend_ts = st.checkbox("Show legend", value=False, key="ts_legend")
    df_plot = convert_unit(df, series_choice, unit1)
    fig1, ax1 = plt.subplots(figsize=(12,4))
    for c in series_choice:
        ax1.plot(df["開始日時"], df_plot[c], label=("After loss" if c.endswith("後") else "Before loss"))
    ax1.set_title("Time Series (All)")
    ax1.set_xlabel("Timestamp"); ax1.set_ylabel("Power [kW]" if unit1=="kW" else "Energy [kWh]")
    ax1.grid(True)
    if show_legend_ts: ax1.legend()
    st.pyplot(fig1, clear_figure=True)

    # downloads
    xls = to_excel_bytes({"TimeSeries": pd.concat([df["開始日時"], df_plot[series_choice]], axis=1)})
    st.download_button("Download Time Series (Excel)", data=xls, file_name="timeseries.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_ts_xlsx")
    st.download_button("Download Time Series (CSV)", data=to_csv_bytes(pd.concat([df["開始日時"], df_plot[series_choice]], axis=1)),
                       file_name="timeseries.csv", mime="text/csv", key="dl_ts_csv")

# ---- Section 2: Distribution ----
st.subheader("Distribution")
if df.empty:
    st.warning("Upload data to view Distribution.")
else:
    target = st.selectbox("Target", ["使用電力量_ロス後","使用電力量_ロス前"], key="dist_target")
    unit2 = st.radio("Unit", ["kWh","kW"], horizontal=True, key="dist_unit")
    bins = st.number_input("Bins", min_value=10, max_value=200, value=50, step=5, key="dist_bins")
    show_legend = st.checkbox("Show legend", value=False, key="dist_legend")

    vec = to_numeric_safe(df[target]).dropna()
    if unit2=="kW": vec = vec * 2.0

    if vec.empty:
        st.warning("Target series has no numeric data.")
    else:
        fig2, ax2 = plt.subplots(figsize=(10,4))
        n, bins_edges, _ = ax2.hist(vec, bins=int(bins))
        mu = np.median(vec)
        sigma = np.std(vec)
        # vertical lines: median and ±σ bands
        for v, lab in [(mu, "Median"), (mu-sigma, "-1σ"), (mu+sigma, "+1σ"), (mu-2*sigma, "-2σ"), (mu+2*sigma, "+2σ"), (mu-3*sigma, "-3σ"), (mu+3*sigma, "+3σ")]:
            ax2.axvline(v, linestyle="--")
        ax2.set_title("Distribution")
        ax2.set_xlabel("Power [kW]" if unit2=="kW" else "Energy [kWh]"); ax2.set_ylabel("Count")
        ax2.grid(True)
        if show_legend:
            ax2.legend([
                f"Median={mu:.1f}", 
                f"-1σ={mu-sigma:.1f}", f"+1σ={mu+sigma:.1f}",
                f"-2σ={mu-2*sigma:.1f}", f"+2σ={mu+2*sigma:.1f}",
                f"-3σ={mu-3*sigma:.1f}", f"+3σ={mu+3*sigma:.1f}",
            ], ncol=2, fontsize="small")
        st.pyplot(fig2, clear_figure=True)

        # download data used
        dist_df = pd.DataFrame({"value": vec})
        st.download_button("Download Distribution Data (CSV)", data=to_csv_bytes(dist_df),
                           file_name="distribution.csv", mime="text/csv", key="dl_dist_csv")

# ---- Section 3: Single Day ----
st.subheader("Single Day")
if df.empty:
    st.warning("Upload data to view Single Day.")
else:
    day = st.date_input("Select a date", value=pd.to_datetime(df["開始日時"].iloc[0]).date(), key="single_day")
    unit3 = st.radio("Unit", ["kWh","kW"], horizontal=True, key="sd_unit")
    series_choice = st.multiselect(
        "Series", ["使用電力量_ロス後","使用電力量_ロス前"], default=["使用電力量_ロス後"], key="sd_series"
    )
    band_width = st.number_input("Shaded band half-width (e.g., ±X)", min_value=0, max_value=10000, value=1000, step=100, key="sd_band")
    show_legend = st.checkbox("Show legend", value=False, key="sd_legend")

    df_day = df[df["開始日時"].dt.date == day]
    if df_day.empty:
        st.warning("No records for the selected date.")
    else:
        df_day = convert_unit(df_day, series_choice, unit3)
        df_day["time"] = df_day["開始日時"].dt.strftime("%H:%M")
        fig3, ax3 = plt.subplots(figsize=(10,4))
        for c in series_choice:
            ax3.plot(df_day["time"], df_day[c], label=("After loss" if c.endswith("後") else "Before loss"))
        # shaded band around median of first selected series
        ref = df_day[series_choice[0]].astype(float)
        med = float(np.median(ref.values))
        ax3.axhline(med, linestyle="--")
        ax3.axhline(med + band_width, linestyle=":")
        ax3.axhline(med - band_width, linestyle=":")
        ax3.fill_between(df_day["time"], med-band_width, med+band_width, alpha=0.2)

        ax3.set_title(f"Single Day — {day.isoformat()}")
        ax3.set_xlabel("Time of Day"); ax3.set_ylabel("Power [kW]" if unit3=="kW" else "Energy [kWh]")
        ax3.grid(True)
        if show_legend: ax3.legend()
        st.pyplot(fig3, clear_figure=True)

        # downloads
        xls = to_excel_bytes({"SingleDay": df_day[["開始日時","time"] + series_choice]})
        st.download_button("Download Single Day (Excel)", data=xls, file_name="single_day.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_sd_xlsx")
        st.download_button("Download Single Day (CSV)", data=to_csv_bytes(df_day[["開始日時","time"] + series_choice]),
                           file_name="single_day.csv", mime="text/csv", key="dl_sd_csv")

# ---- Section 4: Daily Overlay (range) ----
st.subheader("Daily Overlay")
if df.empty:
    st.warning("Upload data to view Overlay.")
else:
    min_date = df["開始日時"].dt.date.min()
    max_date = df["開始日時"].dt.date.max()
    start_date, end_date = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="ol_range")
    unit4 = st.radio("Unit", ["kWh","kW"], horizontal=True, key="ol_unit")
    target = st.selectbox("Target", ["使用電力量_ロス後","使用電力量_ロス前"], key="ol_target")
    show_legend = st.checkbox("Show legend", value=False, key="ol_legend")
    band_width_ol = st.number_input("Shaded band half-width for overlay (±X)", min_value=0, max_value=10000, value=1000, step=100, key="ol_band")

    overlay_rows = []
    fig4, ax4 = plt.subplots(figsize=(12,4))

    # Iterate each day in range
    day = start_date
    ndays = 0
    while day <= end_date:
        df_i = df[df["開始日時"].dt.date == day]
        if not df_i.empty:
            ndays += 1
            df_i = convert_unit(df_i, [target], unit4)
            df_i["time"] = df_i["開始日時"].dt.strftime("%H:%M")
            ax4.plot(df_i["time"], df_i[target], label=day.isoformat())
            overlay_rows.append(df_i.assign(day=day.isoformat())[["day","開始日時","time",target]])
        day = day + timedelta(days=1)

    if ndays == 0:
        st.warning("No records in the selected range.")
    else:
        # shaded band around global median across all selected days (using first day's vector if needed)
        cat = pd.concat(overlay_rows, ignore_index=True)
        ref = to_numeric_safe(cat[target]).dropna()
        if not ref.empty:
            med = float(np.median(ref.values))
            ax4.axhline(med, linestyle="--")
            ax4.axhline(med + band_width_ol, linestyle=":")
            ax4.axhline(med - band_width_ol, linestyle=":")
            # Shade
            ax4.fill_between(cat["time"].unique(), med-band_width_ol, med+band_width_ol, alpha=0.15)

        ax4.set_title(f"Daily Overlay — {start_date.isoformat()} to {end_date.isoformat()} ({'After loss' if target.endswith('後') else 'Before loss'})")
        ax4.set_xlabel("Time of Day"); ax4.set_ylabel("Power [kW]" if unit4=="kW" else "Energy [kWh]")
        ax4.grid(True)
        if show_legend: ax4.legend(ncol=2, fontsize="small")
        st.pyplot(fig4, clear_figure=True)

        # Excel/CSV download (overlay)
        cat = cat.rename(columns={"開始日時": "Start Time"})
        xls = to_excel_bytes({"Overlay": cat})
        st.download_button("Download Overlay (Excel)", data=xls, file_name="overlay.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_overlay_xlsx")
        st.download_button("Download Overlay (CSV)", data=to_csv_bytes(cat),
                           file_name="overlay.csv", mime="text/csv", key="dl_overlay_csv")

# ---- Section 5: DB export ----
st.subheader("Database Export")
if df.empty:
    st.info("No data in DB.")
else:
    st.download_button("Download FULL DB (Excel)", data=to_excel_bytes({"DB": df}),
                       file_name="db_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_db_xlsx")
    st.download_button("Download FULL DB (CSV)", data=to_csv_bytes(df),
                       file_name="db_export.csv", mime="text/csv", key="dl_db_csv")

st.caption("v1.8: added shaded bands with selectable width, CSV downloads, and full DB export.")
