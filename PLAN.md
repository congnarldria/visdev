# 改進計畫

C# 已經能做讀圖、二值化、依距離連通、篩選、矩形形態學、補洞、聯集差集、個數和重心，並從檔案選單匯出。Windows 仍用 `app/engine/visdev_match.pyd`。Linux 的 `visdev_match.so` 不進版控。

## 已完成

`dotnet/Vizion.Operators` 的連通距離 2 以上，用和 OpenCV `distanceTransform` 5×5 相同的權重把靠得近的物件併起來。距離小於 2 仍是八連通。

同一套行程還能做矩形膨脹、侵蝕、開運算、閉運算、補洞、補集、聯集、交集、差集、接起來、按索引取出、用面積或外接矩形篩選、外接矩形、灰階、點是否在區域裡、平均亮度。彩色轉灰階和門檻的像素沒有改。

檔案選單「匯出 C#」會下載 `Run(ViewShow)`。範例是 `samples/export_parts.viz` 和 `samples/export_distance.viz`。用到還沒做的算子時，會指出是哪一行。

2000×1500 的形狀搜尋在沒有 `.pyd` 時約 170 ms，分數仍是 0.9875788688659668。有 `.pyd` 的匹配路徑沒有改。

## 接著做

1. **用區域限制影像。** `reduce_domain` 之後，二值化只處理區域裡面。C# 的影像目前還沒有定義域，所以這條還不能匯出。
2. **標定仍只有一台相機，也不讀描述檔。** `'5,7,10'` 可用，`board.cpd` 會被拒絕。

Windows 執行檔仍由 `build_windows.bat` 在使用者的電腦上產出。這台不是 Windows，沒有產生 exe。
