# Scala (2008) 換投模型：重現與檢驗

把 Natalie M. Scala 的論文 *Decision Modeling and Applications to Major League Baseball Pitcher Substitution*
（29th ASEM National Conference, 2008）裡的五個模型全部寫成可執行的程式，用 2006 年 MLB 真實資料重跑、
做全季回測，並提供換投預測工具。

*English: a runnable reproduction of the five models in Scala (2008) — influence diagram, literature-frequency
criteria weights, Pugh chart, AHP, and a Hirotsu & Wright–style Markov benchmark — with a walk-forward backtest on
the 2006 MLB season and a command-line tool that recommends a pitching change for any 2006 plate appearance.*

## 結論

| 檢驗 | AHP（論文模型） | Markov 基準 | 對照 |
|---|---|---|---|
| 重現論文案例（論文答案：換 Davis） | 建議 Westbrook 續投 | 建議 Westbrook 續投 | 總教練實際換 Davis |
| 「要不要換」AUC（0.5 = 丟硬幣） | 0.588 | 0.595 | 只看先發用球數 0.804 |
| 「換誰」與總教練相同 | 26.2% | 23.3% | 亂猜 18.5%、左右+ERA 法則 28.0% |
| 比較矩陣不合格（CR > 0.10） | 84% 的決策點 | — | — |
| 每次決策計算時間 | 1.8 ms | 12.9 ms | — |

- AHP 的答案主要由論文**沒有公開**的逐項評分決定；把「算平均」的門檻放寬一倍，案例答案就變成 Davis。
- Markov 模型有機率意義、校準良好，但只看本半局，會把最好的牛棚（終結者）提早用掉。
- 完整說明：[`docs/分析報告.md`](docs/分析報告.md)、互動網頁 [`docs/視覺化報告.html`](docs/視覺化報告.html)（下載後用瀏覽器開啟）、
  從零開始的白話說明書 [`docs/白話說明書.docx`](docs/白話說明書.docx)。

## 快速開始

需要 Python 3.10 以上（在 3.12 測過）。

```bash
git clone https://github.com/rex941028/scala2008-pitcher-substitution.git
cd scala2008-pitcher-substitution
pip install -r requirements.txt

python -m unittest discover -s tests -v      # 19 個測試，約 1 分鐘
python predict.py CLE200604250 --inning 6 --half top   # 論文的例子
```

儲存庫已附上處理好的資料（`data/processed/*.csv.gz` 與球員名冊），**不需要另外下載或安裝 Chadwick** 就能跑。

## 換投預測：`predict.py`

對 2006 年任一場比賽的任一個打席，同時用 AHP 和 Markov 兩個模型建議「續投或換誰」。
只用該場比賽**之前**可得的資料（2005 全季 + 2006 當日以前），不偷看未來。

```bash
python predict.py CLE200604250 --list                     # 列出可預測的打席與編號
python predict.py CLE200604250 --index 37                 # 用編號預測
python predict.py CLE200604250 --inning 6 --half top --only westj001,davij005,saues001
python predict.py CLE200604250 --inning 6 --half top --json   # 輸出 JSON
```

比賽代碼是 Retrosheet 格式：主隊代碼 + 日期 + 當天第幾場（例如 `CLE200604250` = 2006/4/25 克里夫蘭主場第 1 場），
球員代碼也是 Retrosheet 格式（`data/raw/2006eve/*.ROS` 可查）。輸出範例：

```
6 局上　0 出局　壘上無人　防守方領先2　打者 Mike Lowell（R）　場上投手 Jake Westbrook
投手                    投     AHP 權重   Markov 勝率     期望失分     零失分　Pugh 小計（狀態/打者/投手/牛棚）
Jake Westbrook（場上）    R      0.422       80.1%     0.48   74.5%　-1 / -2 / +2 / +3
Scott Sauerbeck       L      0.335       78.8%     0.58   71.5%　+1 / -1 / -2 / -2
Jason Davis           R      0.243       79.0%     0.56   72.3%　+0 / -2 / +1 / +0

AHP 建議：Jake Westbrook（續投）
Markov 建議：Jake Westbrook（續投）
實際：Jason Davis（換投），這半局之後失 1 分
```

論文假設有指定打擊（DH），所以國聯主場的比賽結果僅供參考。

## 重現三個實驗

| 指令 | 內容 | 時間 |
|---|---|---|
| `python experiments/e1_case_study.py` | E1：重現論文案例（2006-04-25 BOS @ CLE） | 約 1 分鐘 |
| `python experiments/e2_sensitivity.py` | E2：權重、評分規則、排序反轉、一致性的敏感度 | 約 1 分鐘 |
| `python experiments/e3_backtest.py` | E3：2006 全季回測（AL 主場、第 6 局起每個打席，約 3.8 萬個決策點）；加數字只跑前 N 場 | 6～15 分鐘 |
| `python experiments/e3_analyze.py` | E3 統計分析（依比賽重抽 1,000 次的信賴區間） | 約 2 分鐘 |

輸出在 `results/`（已附上本研究跑出的版本）。

## 專案結構

| 路徑 | 內容 |
|---|---|
| `src/influence_diagram.py` | **模型 1** 影響圖（Exhibit 1，38 個節點逐條箭頭轉錄）與結構檢查 |
| `src/ahp_core.py` | **模型 2** AHP 引擎：特徵向量權重、λmax、CI/CR、distributive / ideal 合成 |
| `src/criteria_weights.py` | **模型 3** 文獻頻率權重（Exhibit 2）與比例 → Saaty 1–9 換算 |
| `src/pugh_chart.py` | **模型 4** Pugh chart：+1/0/−1 評分、總分差 → 比較矩陣 |
| `src/pitcher_ahp.py` | 模型 1–4 組成的 AHP 換投模型（32 個子準則的評分規則） |
| `src/markov_model.py` | **模型 5（基準）** Hirotsu & Wright 式 Markov 鏈：24 壘包出局狀態、左右對決、DERA、勝率表 |
| `src/player_stats.py` | 不偷看未來的球員數據與收縮估計 |
| `src/situation.py` | 逐打席還原總教練當下知道的資訊、推估可用牛棚 |
| `src/context.py`、`src/common.py` | 共用載入與工具 |
| `src/data_prep.py` | 從零重建資料（下載 Retrosheet、Chadwick 轉檔） |
| `predict.py` | 換投預測工具 |
| `experiments/` | E1–E3 實驗 |
| `tests/` | 單元測試（`test_models.py` 不需資料；`test_data_models.py` 用附帶的資料） |
| `results/` | 實驗輸出 |
| `docs/` | 分析報告、互動網頁、白話說明書 |

## 哪些是論文的、哪些是補上的

- **論文有公開：**影響圖結構、比賽狀態／預定打者兩群的文獻頻率、比例換算的例子（0.667 → 1/3）、Pugh 評分方式、案例結論。
- **論文沒公開、本研究補上**（原始資料在未公開的技術報告 Scala 2008, Pitt IE TR 08-1）：
  - 場上投手／牛棚兩群的權重拆分（已知合計 16.5；基準取 10／6.5，E2 掃過全部拆法）；
  - 比例 → 1–9 分的一般規則（取能重現論文例子的規則，另附兩種）；
  - 32 個子準則各自的 +1/0/−1 判定規則（見 `src/pitcher_ahp.py` 開頭）。
- Markov 是簡化版 Hirotsu & Wright：只算到本半局結束，再接聯盟平均勝率表，不是他們 143 萬狀態的整場動態規劃。

## 資料

逐球資料來自 [Retrosheet](https://www.retrosheet.org/)（2005、2006 年），以 [Chadwick](https://github.com/chadwickbureau/chadwick)
的 `cwevent` 轉成 CSV。若要重建或加入其他年份：`python src/data_prep.py --years 2005 2006`
（Windows 會自動下載 Chadwick；macOS 請先 `brew install chadwick`，Linux 請自行編譯並把 `cwevent` 放進 PATH）。
目前的模型以 2005 年為先驗、預測 2006 年。

> The information used here was obtained free of charge from and is copyrighted by Retrosheet.
> Interested parties may contact Retrosheet at "www.retrosheet.org".

論文 PDF 受著作權保護，不放在儲存庫內；可從 [D-Scholarship@Pitt](https://d-scholarship.pitt.edu/concern/generic_works/b658d06a-9611-4710-9411-26ada5b4ad6e)
取得，或執行 `python src/data_prep.py --paper` 從 Internet Archive 下載。

## 引用

Scala, N. M. (2008). Decision Modeling and Applications to Major League Baseball Pitcher Substitution.
*Proceedings of the 29th Annual National Conference of the American Society for Engineering Management.*

## 授權

程式碼以 [MIT License](LICENSE) 釋出。Retrosheet 資料依 Retrosheet 的條款使用（見上方聲明）。
