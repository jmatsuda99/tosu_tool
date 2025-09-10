
import os
import sqlite3
import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ===== 日本語フォント設定 =====
# 実行環境に応じて利用可能な日本語フォントを優先的に指定
matplotlib.rcParams['font.family'] = ['Noto Sans CJK JP', 'IPAexGothic', 'MS Gothic', 'Hiragino Sans', 'Yu Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="使用電力量ビューア（DB保存対応）", layout="wide")
st.title("使用電力量 時系列ビューア（ロス前／ロス後）")
st.caption("X軸：開始日時、Y軸：使用電力量（kWh）。ロス前／ロス後を別々に表示。アップロード済データはDBに保存し、次回以降はファイル無しで利用可能。")

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
    # 期待列チェック
    expected_cols = ["開始日時", "使用電力量(ロス後)", "使用電力量(ロス前)"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.warning(f"必要な列が見つかりません: {missing}")
    # 列名正規化（DB格納用にスペースや括弧を避ける）
    df = df.rename(columns={"使用電力量(ロス後)": "使用電力量_ロス後", "使用電力量(ロス前)": "使用電力量_ロス前"})
    # 日時パース
    if "開始日時" in df.columns:
        df["開始日時"] = pd.to_datetime(df["開始日時"], errors="coerce")
        df = df.dropna(subset=["開始日時"])
    # 必要列に限定
    keep = ["開始日時", "使用電力量_ロス後", "使用電力量_ロス前"]
    df = df[[c for c in keep if c in df.columns]].copy()
    return df

def save_to_db(df):
    # 文字列で保存（ISO形式）
    df2 = df.copy()
    df2["開始日時"] = df2["開始日時"].dt.tz_localize(None) if str(df2["開始日時"].dtype).startswith("datetime64") and df2["開始日時"].dt.tz is not None else df2["開始日時"]
    df2["開始日時"] = pd.to_datetime(df2["開始日時"], errors="coerce")
    df2 = df2.dropna(subset=["開始日時"])
    df2["開始日時"] = df2["開始日時"].dt.strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as conn:
        # 既存データがあればマージ（同一キーは更新）
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

# --- UI: データ入力 ---
init_db()

left, right = st.columns([2,1], gap="large")

with left:
    uploaded = st.file_uploader("Excelファイル（.xlsx）を選択（初回のみ。以後はDBから自動読み込み）", type=["xlsx"])
    if uploaded is not None:
        df_up = load_excel(uploaded)
        if not df_up.empty:
            save_to_db(df_up)
            st.success(f"アップロードデータをDBへ保存しました。レコード数：{len(df_up):,}")
        else:
            st.warning("アップロードされたデータに有効な行が見つかりませんでした。")

with right:
    if st.button("DBをクリア（削除）", type="secondary"):
        clear_db()
        st.warning("DBをクリアしました。再読み込みしてください。")

# --- データ読み込み（DB） ---
df = load_from_db()
if df.empty:
    # 既定パスのデモデータを初回ブートストラップとして読み込むオプション
    default_path = "/mnt/data/240801 24年8月～25年7月鳥栖PO1期.xlsx"
    if os.path.exists(default_path):
        st.info("DBが空だったため、既定パスのファイルを読み込みDBに保存しました。")
        df_demo = load_excel(default_path)
        if not df_demo.empty:
            save_to_db(df_demo)
            df = load_from_db()
    else:
        st.stop()

# --- 表示粒度 ---
view = st.radio("表示粒度", ["30分（そのまま）", "日別合計", "月別合計"], horizontal=True)

plot_df = df.copy()
y_cols = ["使用電力量_ロス後", "使用電力量_ロス前"]
plot_df = plot_df[["開始日時"] + [c for c in y_cols if c in plot_df.columns]].copy()

if view == "30分（そのまま）":
    display_df = plot_df.sort_values("開始日時")
elif view == "日別合計":
    display_df = (
        plot_df.set_index("開始日時")
               .resample("D")
               .sum(numeric_only=True)
               .reset_index()
    )
    display_df.rename(columns={"開始日時": "日付"}, inplace=True)
elif view == "月別合計":
    display_df = (
        plot_df.set_index("開始日時")
               .resample("MS")
               .sum(numeric_only=True)
               .reset_index()
    )
    display_df.rename(columns={"開始日時": "年月"}, inplace=True)

# --- グラフ ---
fig, ax = plt.subplots(figsize=(10, 4))
x_col = "開始日時" if view == "30分（そのまま）" else ("日付" if view == "日別合計" else "年月")
for col in [c for c in y_cols if c in display_df.columns]:
    ax.plot(display_df[x_col], display_df[col], label=col)

ax.set_xlabel(x_col)
ax.set_ylabel("使用電力量 [kWh]")
ax.set_title("使用電力量（ロス後／ロス前）")
ax.grid(True)
ax.legend()
st.pyplot(fig, clear_figure=True)

# --- 表示中データとCSV ---
st.subheader("表示中データ（先頭50行）")
st.dataframe(display_df.head(50))

csv = display_df.to_csv(index=False).encode("utf-8-sig")
st.download_button("表示中データをCSVでダウンロード", data=csv, file_name="display_data.csv", mime="text/csv")

st.caption("※ 日本語フォントは実行環境の有無に依存します（Noto Sans CJK JPなど）。")
