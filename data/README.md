# 資料 / Data

| 檔案 | 內容 |
|---|---|
| `processed/events_2005.csv.gz`、`processed/events_2006.csv.gz` | Retrosheet 逐球紀錄經 Chadwick `cwevent` 轉成的 CSV（欄位見 `src/data_prep.py`） |
| `raw/2005eve/*.ROS`、`raw/2006eve/*.ROS` | Retrosheet 球員名冊（左右投打） |

重建：`python src/data_prep.py --years 2005 2006`（會下載 Retrosheet 原始檔到 `raw/`，原始檔不放進 git）。

> The information used here was obtained free of charge from and is copyrighted by Retrosheet.
> Interested parties may contact Retrosheet at "www.retrosheet.org".
