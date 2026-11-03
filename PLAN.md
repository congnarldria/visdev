# 改進計畫

從 VisDev 1.1 beta 接著做。這還不是正式版。公差先停在這裡。Windows 仍用 `app/engine/visdev_match.pyd`。開相機仍先問 DirectShow，打不開再試 OpenCV。Linux 的 `visdev_match.so` 不進版控。

## 已完成

產線範本用 `samples/line_batch` 的 5 張圖模擬連續取像。白塊面積 200 以上算合格。合格 3、不合格 2。不合格的圖寫到 `workspace/line_ng`，明細寫到 `workspace/line_result.csv`。狀態列寫「判定 不合格 · 合格 3 · 不合格 2 · 總數 5」。這支用 File 資料夾，不開相機。

卡尺、條紋、瑕疵和公差都還在。單台相機的標定數字沒有改。版本仍是 1.1 beta。

## 接著做

1. **產線改接本機相機，按停止就停。** 資料夾模式已經能計數、存 NG 圖、寫 CSV。下一步用 OpenCV 相機連續抓圖，沿用同一套合格規則。Windows 仍先問 DirectShow。按停止時留下已經計的張數，不必等迴圈自己結束。

## 以後

六角形標定板的尋找標記對位還沒做。`'5,7,10'` 和矩形圓點板可以用。
