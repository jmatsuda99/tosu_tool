
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

APP_TITLE = "Energy Viewer v1.8 — all-days totals + robust reload"
DB_PATH = "energy_cache.db"
TABLE_NAME = "energy_records"

st.set_page_config(page_title="Energy Viewer v1.8", layout="wide")
st.title(APP_TITLE)

# ---- Utilities ----
def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def db_mtime() -> float:
    try:
        return os.path.getmtime(DB_PATH)
    except FileNotFoundError:
        return 0.0

@st.cache_data(show_spinner=False)
def load_from_db(mtime_tag: float) -> pd.DataFrame:
    """Load DB with cache keyed by mtime_tag so external DB changes invalidate cache."""
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
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("Clear DB"):
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        st.sidebar.success("Database cleared.")
        st.cache_data.clear()
with col_btn2:
    if st.button("Reload data"):
        # Bump cache by calling load with new mtime tag
        st.cache_data.clear()

if file is not None:
    try:
        if file.name.lower().endswith(".csv"):
            src = pd.read_csv(file)
        else:
            src = pd.read_excel(file)
        # === 最小前処理：列名変換＆必要列のみ抽出 ===
        src.columns = [str(c).strip() for c in src.columns]
        rename_map = {"使用電力量(ロス後)": "使用電力量_ロス後","使用電力量(ロス前)": "使用電力量_ロス前"}
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

# load with mtime-dependent cache key
df = load_from_db(db_mtime())
st.caption(f"Records in DB: {len(df)}")

# ---- Section 1: Time Series ----
st.subheader("Time Series")
if df.empty:
    st.info("Upload data to view charts.")
else:
    unit1 = st.radio("Unit", ["kWh","kW"], horizontal=True, key="ts_unit")
    series_choice = st.multiselect("Series", ["使用電力量_ロス後","使用電力量_ロス前"], default=["使用電力量_ロス後"], key="ts_series")
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
        for k in [-3,-2,-1,0,1,2,3]:
            ax2.axvline(mu + k*sigma, linestyle="--")
        ax2.set_title("Distribution")
        ax2.set_xlabel("Power [kW]" if unit2=="kW" else "Energy [kWh]"); ax2.set_ylabel("Count")
        ax2.grid(True)
        if show_legend:
            labels = [f"{k:+d}σ={mu+k*sigma:.1f}" if k!=0 else f"Median={mu:.1f}" for k in [0,-1,1,-2,2,-3,3]]
            ax2.legend(labels, ncol=2, fontsize="small")
        st.pyplot(fig2, clear_figure=True)

# ---- Section 3: Single Day (FULL) ----
st.subheader("Single Day")
if df.empty:
    st.warning("Upload data to view Single Day.")
else:
    day = st.date_input("Select a date", value=pd.to_datetime(df["開始日時"].iloc[0]).date(), key="single_day")
    unit3 = st.radio("Unit", ["kWh","kW"], horizontal=True, key="sd_unit")
    series_choice = st.multiselect("Series", ["使用電力量_ロス後","使用電力量_ロス前"], default=["使用電力量_ロス後"], key="sd_series")
    band_width = st.number_input("Shaded band half-width (e.g., ±X)", min_value=0, max_value=10000, value=1000, step=100, key="sd_band")
    show_legend = st.checkbox("Show legend", value=False, key="sd_legend")
    show_bars = st.checkbox("Show deviation bars (before loss only) + totals", value=True, key="sd_bars")

    df_day = df[df["開始日時"].dt.date == day]
    if df_day.empty:
        st.warning("No records for the selected date.")
    else:
        df_day = convert_unit(df_day, series_choice, unit3)
        df_day["time"] = df_day["開始日時"].dt.strftime("%H:%M")
        fig3, ax3 = plt.subplots(figsize=(12,5))

        # 3-hour mean and deviations for selected series (for dashed lines)
        df_idx = df_day.set_index("開始日時")
        three_h_mean_any = {}
        for c in series_choice:
            s = to_numeric_safe(df_idx[c])
            m = s.resample("3H").mean()
            aligned = m.reindex(df_idx.index, method="ffill")
            three_h_mean_any[c] = aligned.values

        # Actual lines
        for c in series_choice:
            ax3.plot(df_day["time"], df_day[c], label=("After loss" if c.endswith("後") else "Before loss"))
        # 3h mean dashed lines
        for c in series_choice:
            ax3.plot(df_day["time"], three_h_mean_any[c], linestyle="--", label=f"{'After loss' if c.endswith('後') else 'Before loss'} (3h mean)")

        # Median ± band (based on first selected series if available)
        if series_choice:
            ref = df_day[series_choice[0]].astype(float)
            med = float(np.median(ref.values))
            ax3.axhline(med, linestyle="--")
            ax3.axhline(med + band_width, linestyle=":")
            ax3.axhline(med - band_width, linestyle=":")
            ax3.fill_between(df_day["time"], med-band_width, med+band_width, alpha=0.2)

        # Deviation bars for BEFORE LOSS only (使用電力量_ロス前)
        if show_bars and "使用電力量_ロス前" in df_idx.columns:
            s = to_numeric_safe(df_idx["使用電力量_ロス前"])
            m = s.resample("3H").mean()
            aligned = m.reindex(df_idx.index, method="ffill")
            deviation = aligned.values - s.values  # signed (mean - actual)

            # Draw bars anchored at mean
            x = np.arange(len(df_day))
            width = 0.5
            heights = -deviation  # + -> down (negative height), - -> up (positive height)
            colors = ["tab:blue" if d > 0 else "tab:red" if d < 0 else "0.5" for d in deviation]
            ax3.bar(x, heights, bottom=aligned.values, width=width, color=colors, alpha=0.35, linewidth=0)

            # Compute daily totals in kWh
            if unit3 == "kW":
                blue_total_kwh = float(np.sum(np.clip(deviation, 0, None)) * 0.5)
                red_total_kwh  = float(np.sum(np.clip(-deviation, 0, None)) * 0.5)
            else:
                blue_total_kwh = float(np.sum(np.clip(deviation, 0, None)))
                red_total_kwh  = float(np.sum(np.clip(-deviation, 0, None)))

            # Put totals in the chart (bottom-right)
            txt = f"Blue total: {blue_total_kwh:.1f} kWh\nRed total: {red_total_kwh:.1f} kWh"
            ax3.text(0.99, 0.02, txt, transform=ax3.transAxes, ha="right", va="bottom",
                     bbox=dict(boxstyle="round,pad=0.4", alpha=0.2))

        ax3.set_title(f"Single Day — {day.isoformat()} ({'kW' if unit3=='kW' else 'kWh'})")
        ax3.set_xlabel("Time of Day"); ax3.set_ylabel("Power [kW]" if unit3=="kW" else "Energy [kWh]")
        ax3.grid(True)
        if show_legend: ax3.legend(ncol=2, fontsize="small")
        st.pyplot(fig3, clear_figure=True)

        # Deviation-only chart
        if show_bars and "使用電力量_ロス前" in df_idx.columns:
            fig3d, ax3d = plt.subplots(figsize=(12,3.6))
            s = to_numeric_safe(df_idx["使用電力量_ロス前"])
            m = s.resample("3H").mean()
            aligned = m.reindex(df_idx.index, method="ffill")
            deviation = aligned.values - s.values
            ax3d.plot(df_day["time"], deviation, label="Before loss deviation")
            ax3d.axhline(0.0, linestyle="--")
            ax3d.set_title("Deviation from 3h Mean (mean - actual) — Before loss")
            ax3d.set_xlabel("Time of Day"); ax3d.set_ylabel("Δ [kW]" if unit3=="kW" else "Δ [kWh]")
            ax3d.grid(True)
            if show_legend: ax3d.legend(ncol=2, fontsize="small")
            st.pyplot(fig3d, clear_figure=True)

            dev_df = pd.DataFrame({"time": df_day["time"], "deviation": deviation})
            st.download_button("Download Deviation (Before loss) CSV", data=to_csv_bytes(dev_df),
                               file_name="single_day_deviation_before_loss.csv", mime="text/csv", key="dl_sd_dev_b")

# ---- Section 4: All Days Totals (Before loss) ----
st.subheader("All Days Totals — Blue/Red energy sums (Before loss)")
if df.empty or "使用電力量_ロス前" not in df.columns:
    st.warning("Before loss データがありません。アップロードを確認してください。")
else:
    # always compute from current df to reflect latest DB (df is from mtime-aware cache)
    df2 = df.copy()
    df2["使用電力量_ロス前"] = to_numeric_safe(df2["使用電力量_ロス前"]) * 2.0  # kW
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

    c1, c2 = st.columns([2,1])
    with c1:
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

    st.download_button("Download All-Days Totals (CSV)", data=to_csv_bytes(daily_df),
                       file_name="all_days_totals.csv", mime="text/csv", key="dl_all_days_csv")

st.caption("Reload-safe: DB更新時は自動で再読込（mtimeキー）。Single Dayはフル機能、All Days合計は現行DBから常時計算。")
