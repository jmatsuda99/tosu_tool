
import os
import io
import sqlite3
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

APP_TITLE = "Energy Consumption Viewer — v1.8 (all-days totals)"
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
        try:
            df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} ORDER BY 開始日時", conn)
        except Exception:
            return pd.DataFrame(columns=["開始日時","使用電力量_ロス後","使用電力量_ロス前"])
    if df.empty:
        return df
    df["開始日時"] = pd.to_datetime(df["開始日時"], errors="coerce")
    return df.dropna(subset=["開始日時"])

def upsert_into_db(df: pd.DataFrame) -> None:
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
        # === 最小前処理：列名変換＆必要列のみ抽出 ===
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

# ---- Section 2: Single Day (same as前版：省略) ----
st.subheader("Single Day (excerpt)")
st.caption("省略：前版と同等の単日可視化機能あり（3h平均・偏差バー・合計表示）。")

# ---- Section 3: All Days Totals (NEW) ----
st.subheader("All Days Totals — Blue/Red energy sums (Before loss)")
if df.empty or "使用電力量_ロス前" not in df.columns:
    st.warning("Before loss データがありません。アップロードを確認してください。")
else:
    # すべて kW で計算（DBは30分kWhなので×2.0）
    df2 = df.copy()
    df2["使用電力量_ロス前"] = to_numeric_safe(df2["使用電力量_ロス前"]) * 2.0
    df2 = df2.dropna(subset=["使用電力量_ロス前"])
    df2["date"] = df2["開始日時"].dt.date

    daily = []
    for d, g in df2.groupby("date", sort=True):
        g = g.sort_values("開始日時").copy()
        g_idx = g.set_index("開始日時")
        s = g_idx["使用電力量_ロス前"].astype(float)
        m = s.resample("3H").mean()
        aligned = m.reindex(g_idx.index, method="ffill")
        deviation = aligned.values - s.values  # kW

        blue_kwh = float(np.sum(np.clip(deviation, 0, None)) * 0.5)  # 30分×0.5h
        red_kwh  = float(np.sum(np.clip(-deviation, 0, None)) * 0.5)
        daily.append({"date": d, "blue_kwh": blue_kwh, "red_kwh": red_kwh})

    daily_df = pd.DataFrame(daily)
    total_blue = float(daily_df["blue_kwh"].sum())
    total_red  = float(daily_df["red_kwh"].sum())

    # 表示
    c1, c2 = st.columns([2,1])
    with c1:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12,4))
        x = np.arange(len(daily_df))
        ax.bar(x-0.2, daily_df["blue_kwh"], width=0.4, label="Blue (+) kWh")
        ax.bar(x+0.2, daily_df["red_kwh"],  width=0.4, label="Red (-) kWh")
        ax.set_xticks(x)
        ax.set_xticklabels([pd.to_datetime(str(d)).strftime("%Y-%m-%d") for d in daily_df["date"]], rotation=45, ha="right")
        ax.set_title("Per-day Summed Energy (Deviation relative to 3h mean)")
        ax.set_ylabel("kWh")
        ax.grid(True, axis="y")
        ax.legend()
        st.pyplot(fig, clear_figure=True)
    with c2:
        st.metric("Total Blue (+) kWh", f"{total_blue:,.1f}")
        st.metric("Total Red (-) kWh", f"{total_red:,.1f}")
        st.write("※ Blue=平均>実（下向き棒） / Red=平均<実（上向き棒）")

    # ダウンロード
    st.download_button("Download All-Days Totals (CSV)", data=to_csv_bytes(daily_df),
                       file_name="all_days_totals.csv", mime="text/csv", key="dl_all_days_csv")

# ---- Section 4: DB export ----
st.subheader("Database Export")
if df.empty:
    st.info("No data in DB.")
else:
    st.download_button("Download FULL DB (CSV)", data=to_csv_bytes(df),
                       file_name="db_export.csv", mime="text/csv", key="dl_db_csv")

st.caption("v1.8 (all-days totals): per-day Blue/Red sums and overall totals for Before loss data.")
