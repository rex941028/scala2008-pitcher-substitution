# E2：敏感度分析（論文案例 2006-04-25，候選 = Westbrook / Davis / Sauerbeck）

## A. 未公開的 Pitcher/Bullpen 權重拆分（兩者和固定 16.5）

每種換算規則下，34 種拆分中各投手勝出的次數：

| rule | Sauerbeck | Westbrook |
|---|---|---|
| exact | 0 | 34 |
| paper | 2 | 32 |
| round | 1 | 33 |

- paper: Westbrook: Pitcher=0.0~8.5；Sauerbeck: Pitcher=9.0~9.5；Westbrook: Pitcher=10.0~16.5

- round: Westbrook: Pitcher=0.0~15.0；Sauerbeck: Pitcher=15.5~15.5；Westbrook: Pitcher=16.0~16.5

- exact: Westbrook: Pitcher=0.0~16.5

## B. 準則權重隨機擾動（Dirichlet，平均 = 文獻頻率權重，集中度 20，5000 次）

| 投手 | 勝出比例 |
|---|---|
| Westbrook | 71.7% |
| Davis | 0.0% |
| Sauerbeck | 28.3% |

在整個權重單純形上（步長 0.05，1757 組），Davis 勝出的組合佔 0.0%。

## C. 評分規則的一次一改（論文未公開逐項評分，這些都是合理讀法）

| 變體 | Westbrook | Davis | Sauerbeck | 建議 |
|---|---|---|---|---|
| 基準（本研究的評分規則） | 0.422 | 0.243 | 0.335 | Westbrook |
| 場上投手在 Bullpen 準則全給 0（31/36/37 不給 +1） | 0.382 | 0.278 | 0.339 | Westbrook |
| 疲勞門檻改為用球 ≥ 平常 90% 即 -1 | 0.422 | 0.243 | 0.335 | Westbrook |
| 今日表現只看失分（2 分 → 0） | 0.455 | 0.214 | 0.331 | Westbrook |
| State 情境拆分全給 0（拆分樣本太小視為無法判定） | 0.451 | 0.252 | 0.297 | Westbrook |
| 板凳代打節點 3–6 全給 0 | 0.508 | 0.329 | 0.164 | Westbrook |
| Westbrook 用球 98 + 被上壘 .500 → 疲勞與今日表現都 -1 且 Bullpen 給 0 | 0.382 | 0.278 | 0.339 | Westbrook |
| Pugh 差距改線性換算（差 1→2、2→3…） | 0.414 | 0.283 | 0.303 | Westbrook |
| 「平均」容忍帶 ×0.5 | 0.422 | 0.327 | 0.251 | Westbrook |
| 「平均」容忍帶 ×2.0 | 0.279 | 0.414 | 0.307 | Davis |

## D. 排序反轉（rank reversal）：在三位候選外再加一位投手

| 加入 | 合成 | 原三人排序 | 加入後三人排序 | 反轉 |
|---|---|---|---|---|
| Rafael Betancourt | distributive | Westbrook > Sauerbeck > Davis | Sauerbeck > Westbrook > Davis | True |
| Rafael Betancourt | ideal | Westbrook > Sauerbeck > Davis | Sauerbeck > Westbrook > Davis | True |
| Danny Graves | distributive | Westbrook > Sauerbeck > Davis | Westbrook > Sauerbeck > Davis | False |
| Danny Graves | ideal | Westbrook > Sauerbeck > Davis | Westbrook > Sauerbeck > Davis | False |
| Jeremy Guthrie | distributive | Westbrook > Sauerbeck > Davis | Westbrook > Sauerbeck > Davis | False |
| Jeremy Guthrie | ideal | Westbrook > Sauerbeck > Davis | Westbrook > Sauerbeck > Davis | False |
| Matt Miller | distributive | Westbrook > Sauerbeck > Davis | Sauerbeck > Westbrook > Davis | True |
| Matt Miller | ideal | Westbrook > Sauerbeck > Davis | Sauerbeck > Westbrook > Davis | True |
| Guillermo Mota | distributive | Westbrook > Sauerbeck > Davis | Westbrook > Sauerbeck > Davis | False |
| Guillermo Mota | ideal | Westbrook > Sauerbeck > Davis | Sauerbeck > Westbrook > Davis | True |
| Rafael Perez | distributive | Westbrook > Sauerbeck > Davis | Westbrook > Sauerbeck > Davis | False |
| Rafael Perez | ideal | Westbrook > Sauerbeck > Davis | Westbrook > Sauerbeck > Davis | False |
| Brian Slocum | distributive | Westbrook > Sauerbeck > Davis | Westbrook > Sauerbeck > Davis | False |
| Brian Slocum | ideal | Westbrook > Sauerbeck > Davis | Westbrook > Sauerbeck > Davis | False |
| Bob Wickman | distributive | Westbrook > Sauerbeck > Davis | Sauerbeck > Westbrook > Davis | True |
| Bob Wickman | ideal | Westbrook > Sauerbeck > Davis | Sauerbeck > Westbrook > Davis | True |

反轉次數：distributive 3/8，ideal 4/8

## E. Pugh 差距換算出的比較矩陣一致性（CR）

| 候選數 | 總分組合數 | CR>0.10 比例 | CR 中位數 | CR 最大 |
|---|---|---|---|---|
| 3 | 729 | 46.9% | 0.069 | 0.376 |
| 4 | 6561 | 39.5% | 0.089 | 0.278 |
| 5 | 59049 | 45.4% | 0.096 | 0.201 |

（每位候選在一個準則下的 Pugh 總分取 -4..+4 的所有組合）

## F. 論文宣稱的兩兩比較次數

| 候選數 | 論文 | 實際（4 準則 6 次 + 每準則 n(n-1)/2） |
|---|---|---|
| 3 | 18 | 18 |
| 4 | 26 | 30 |
| 5 | 46 | 46 |
