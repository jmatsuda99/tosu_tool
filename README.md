
# 使用電力量 時系列ビューア（ロス前／ロス後） v1.1

- `import os` を追加し、未アップロード時の既定パス読み込みで NameError が出ないよう修正しました。
- 機能はv1と同じ：30分/日別/月別の切替、ロス前/ロス後の2系列表示、CSVエクスポート。

## 起動
```bash
pip install -r requirements.txt
streamlit run app.py
```
