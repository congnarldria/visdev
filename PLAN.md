# 改進計畫

從 VisDev 1.1 beta 接著做。這還不是正式版。公差先停在這裡，編輯器的局部更新也先停在這裡。產線的不合格圖瀏覽也先停在這裡。Windows 仍用 `app/engine/visdev_match.pyd`。開相機仍先問 DirectShow，打不開再試 OpenCV。開 COM 仍用原來的 API。Linux 的 `visdev_match.so` 不進版控。

## 已完成

字元範本讀 `samples/ocr_label.png`。`Chars` 是 VISDEV 168，字塊外框畫在圖上。只有一塊、信心至少 0.5 才合格。用本機引擎，不把圖送到雲端。

讀到的字可以另送一行 `VISDEV,TEXT,文字`。產線的 OK、NG 那一行欄位沒有改。色彩的面積 1200、色調 60 沒有改。

## 接著做

1. **合格字可以改。** 字元範本現在寫死 VISDEV 168。改成程式裡的預期字，圖形視窗能改，再按執行就用新的字判斷。不改讀取引擎，也不改 Windows 開相機和開 COM 的順序。

## 以後

六角形標定板的尋找標記對位還沒做。`'5,7,10'` 和矩形圓點板可以用。Modbus 還沒做。
