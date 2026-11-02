# 改進計畫

讀圖、二值化、八連通已經能在 C# 算出和 Python 相同的像素，並從檔案選單匯出。大圖上最慢的是沒有載入 `.pyd` 時的形狀搜尋，同一張 2000×1500 從約 815 ms 降到約 170 ms，分數沒有改。Windows 仍用 `app/engine/visdev_match.pyd`。Linux 的 `visdev_match.so` 不進版控。

## 已完成

`dotnet/Vizion.Operators` 提供 `ReadImage`、`Threshold`、`Connection`、`Count`、`AreaCenterOf`。彩色灰階用和 OpenCV 相同的整數式。門檻含兩端。連通是八連通，物件依掃描到的第一個像素排序，距離 2 以上先不匯出。行程可以直接交給現有的 `VizionWindow.Show`。

檔案選單的「匯出 C#」會下載一支 `Run(ViewShow)`。只含讀圖、二值化、連通、個數、重心和顯示的程式可以匯出；用到別的算子時，會指出是哪一行。範例是 `samples/export_parts.viz`。

2000×1500、約 80 個亮塊，這台沒有載入 `.pyd`：讀圖約 27 ms，二值化約 26 ms，連通約 27 ms（原先約 120 ms），形狀搜尋約 170 ms（原先約 815 ms）。搜尋改的是取樣座標的算法開銷，同一張圖的分數仍是 0.9875788688659668。有 `.pyd` 的匹配路徑沒有改。

更早完成的還有：本機相機每次抓目前畫面、啟動失敗會寫 `visdev.log` 並在 Windows 跳出「VisDev 沒有啟動」、`build_windows.bat`。除錯器可以單步進程序、看遞迴層數。

## 接著做

1. **連通距離 2 以上也做成 C#，並允許匯出。** 現在距離 1 和省略距離可以匯出。距離 5 這種「靠得近就併成一個」會被拒絕。
2. **標定仍只有一台相機，也不讀描述檔。** `'5,7,10'` 可用，`board.cpd` 會被拒絕。

Windows 執行檔仍由 `build_windows.bat` 在使用者的電腦上產出。這台不是 Windows，沒有產生 exe。
