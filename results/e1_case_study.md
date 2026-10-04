# E1：重現論文案例（2006-04-25 BOS @ CLE，6 局上，Lowell 打擊）

- 場上：CLE 領先 2 分，0 出局，壘包代碼 0，打者 Mike Lowell（R），下一棒 Willie Harris（L）
- 板凳：Josh Bard(B), Alex Cora(L), Dustan Mohr(R), Wily Mo Pena(R), J.T. Snow(L), Adam Stern(L)
- Westbrook 今日：98 球、24 打席、上壘 12、失分 2；平常先發用球 96
- 實際：Jason Davis 上場；本半局失 1 分

## 候選投手（賽前可得數據，已做收縮）

| 投手 | 投 | ERA(收縮) | 被打擊率 vs L | vs R | 被上壘率 | DERA vs L | DERA vs R | 休息天數 | 近3日用球 |
|---|---|---|---|---|---|---|---|---|---|
| Jake Westbrook | R | 4.56 | 0.269 | 0.256 | 0.312 | 4.73 | 4.12 | 5 | 0 |
| Jason Davis | R | 4.34 | 0.274 | 0.267 | 0.341 | 5.40 | 4.81 | 1 | 28 |
| Scott Sauerbeck | L | 4.17 | 0.237 | 0.272 | 0.347 | 4.23 | 5.53 | 2 | 9 |

## Pugh chart（本模型的 +1/0/-1 評分）

**State**

| 子準則 | Jake Westbrook | Jason Davis | Scott Sauerbeck |
|---|---|---|---|
| 12. 上/下半局 | +0 | +0 | +0 |
| 13. 局數 | +1 | +0 | +0 |
| 14. 主/客場 | +0 | +0 | +0 |
| 15. 領先/落後分數 | -1 | +0 | +0 |
| 16. 領先或落後 | -1 | +0 | +0 |
| 17. 出局數 | +0 | +0 | +1 |
| 18. 上個半局發生的事 | +0 | +0 | +0 |
| 19. 壘上跑者 | +0 | +0 | +0 |
| Total | -1 | +0 | +1 |

**Batter**

| 子準則 | Jake Westbrook | Jason Davis | Scott Sauerbeck |
|---|---|---|---|
| 3. 板凳打者慣用手 | -1 | -1 | -1 |
| 4. 板凳打者打擊率 | +0 | +0 | +0 |
| 5. 板凳打者左右投偏好 | -1 | -1 | -1 |
| 6. 對手可派的代打 | -1 | -1 | +1 |
| 7. 下一棒打者 | -1 | -1 | +1 |
| 8. 預定打者慣用手 | +1 | +1 | -1 |
| 10. 預定打者左右投偏好 | +0 | +0 | -1 |
| 11. 預定打者打擊率 | +1 | +1 | +1 |
| Total | -2 | -2 | -1 |

**Pitcher**

| 子準則 | Jake Westbrook | Jason Davis | Scott Sauerbeck |
|---|---|---|---|
| 21. 投手慣用手 | +1 | +1 | -1 |
| 22. 投手對左右打防禦率 | +1 | +0 | +0 |
| 23. 疲勞 | -1 | +1 | +1 |
| 24. 是否受傷 | +0 | +0 | +0 |
| 25. 今日表現（失分） | -1 | +0 | +0 |
| 26. 投手防禦率 | +0 | +0 | +0 |
| 27. 投手一般數據 | +1 | -1 | -1 |
| 28. 場上投手 | +0 | +0 | +0 |
| 30. 投手 DERA | +1 | +0 | -1 |
| Total | +2 | +1 | -2 |

**Bullpen**

| 子準則 | Jake Westbrook | Jason Davis | Scott Sauerbeck |
|---|---|---|---|
| 31. 牛棚休息 | +1 | +0 | +1 |
| 32. 牛棚對左右打防禦率 | +0 | +0 | +0 |
| 34. 牛棚防禦率 | +0 | +0 | +0 |
| 35. 牛棚 DERA | +0 | +0 | -1 |
| 36. 先發輪值 | +1 | +1 | -1 |
| 37. 後續賽程 | +1 | +0 | +0 |
| 38. 牛棚一般數據 | +0 | -1 | -1 |
| Total | +3 | +0 | -2 |

## AHP 結果（基準：論文的四捨五入規則、distributive 合成）

準則權重：State 0.125，Batter 0.375，Pitcher 0.375，Bullpen 0.125（CR = 0.000）

| 準則 | Jake Westbrook | Jason Davis | Scott Sauerbeck | CR |
|---|---|---|---|---|
| State | 0.105 | 0.258 | 0.637 | 0.033 |
| Batter | 0.200 | 0.200 | 0.600 | 0.000 |
| Pitcher | 0.633 | 0.304 | 0.063 | 0.117 |
| Bullpen | 0.772 | 0.173 | 0.055 | 0.180 |
| **整體** | **0.422** | **0.243** | **0.335** | |

→ AHP 建議：**Jake Westbrook**（論文：Jason Davis；實際：Jason Davis）

## 不同換算規則／合成方式下的建議

| 規則/合成 | Jake Westbrook | Jason Davis | Scott Sauerbeck | 建議 |
|---|---|---|---|---|
| paper/distributive | 0.422 | 0.243 | 0.335 | Jake Westbrook |
| paper/ideal | 0.410 | 0.244 | 0.347 | Jake Westbrook |
| round/distributive | 0.402 | 0.240 | 0.358 | Jake Westbrook |
| round/ideal | 0.387 | 0.241 | 0.371 | Jake Westbrook |
| exact/distributive | 0.417 | 0.238 | 0.345 | Jake Westbrook |
| exact/ideal | 0.401 | 0.239 | 0.360 | Jake Westbrook |

## Markov 基準模型（Hirotsu & Wright 式）

| 投手 | 本半局期望失分 | 本半局零失分機率 | 防守方勝率 |
|---|---|---|---|
| Jake Westbrook | 0.481 | 0.745 | 0.8006 |
| Jason Davis | 0.560 | 0.723 | 0.7902 |
| Scott Sauerbeck | 0.580 | 0.715 | 0.7877 |

→ Markov 建議：**Jake Westbrook**

## 若把當晚所有可用牛棚都列為候選

| 投手 | 投 | AHP 整體權重 | Markov 防守方勝率 |
|---|---|---|---|
| Jake Westbrook（場上） | R | 0.070 | 0.8006 |
| Rafael Betancourt | R | 0.228 | 0.8092 |
| Jason Davis | R | 0.044 | 0.7902 |
| Danny Graves | R | 0.031 | 0.7807 |
| Jeremy Guthrie | R | 0.056 | 0.7948 |
| Matt Miller | R | 0.116 | 0.7941 |
| Guillermo Mota | R | 0.097 | 0.7934 |
| Rafael Perez | L | 0.090 | 0.7958 |
| Scott Sauerbeck | L | 0.087 | 0.7877 |
| Brian Slocum | R | 0.044 | 0.7924 |
| Bob Wickman | R | 0.136 | 0.8005 |

→ AHP：**Rafael Betancourt**；Markov：**Rafael Betancourt**
