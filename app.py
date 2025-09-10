
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Usage Energy Viewer", layout="wide")

st.title("使用電力量 時系列ビューア（ロス前／ロス後）")
st.write("X軸：開始日時、Y軸：使用電力量（kWh）。ロス前／ロス後を別々に表示します。")

uploaded = st.file_uploader("Excelファイル（.xlsx）を選択（未選択時はデモデータを使用）", type=["xlsx"])

@st.cache_data
def load_excel(file):
    # engineは自動判定（openpyxl前提）
    df = pd.read_excel(file)
    expected_cols = ["開始日時", "使用電力量(ロス後)", "使用電力量(ロス前)"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.warning(f"必要な列が見つかりません: {missing}")
    if "開始日時" in df.columns:
        df["開始日時"] = pd.to_datetime(df["開始日時"], errors="coerce")
        df = df.dropna(subset=["開始日時"])
    return df

default_path = "/mnt/data/240801 24年8月～25年7月鳥栖PO1期.xlsx"
if uploaded is not None:
    df = load_excel(uploaded)
elif os.path.exists(default_path):
    df = load_excel(default_path)
    st.info("デモとして既定パスのファイルを読み込みました。")
else:
    st.stop()

view = st.radio("表示粒度", ["30分（そのまま）", "日別合計", "月別合計"], horizontal=True)

plot_df = df.copy()
y_cols = ["使用電力量(ロス後)", "使用電力量(ロス前)"]
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

st.subheader("表示中データ（先頭50行）")
st.dataframe(display_df.head(50))

csv = display_df.to_csv(index=False).encode("utf-8-sig")
st.download_button("表示中データをCSVでダウンロード", data=csv, file_name="display_data.csv", mime="text/csv")

st.caption("※グラフの日本語フォントが欠ける場合は、OS側の日本語フォント設定に依存します。")
