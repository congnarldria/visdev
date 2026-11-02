# 改進計畫

C# 已經能做圓形膨脹、侵蝕、開運算和閉運算，核和 OpenCV 的橢圓結構元素相同。也能取出區域的內邊界和外邊界。Windows 仍用 `app/engine/visdev_match.pyd`。Linux 的 `visdev_match.so` 不進版控。

## 已完成

`DilateCircle`、`ErodeCircle`、`OpenCircle`、`CloseCircle` 作用在區域聯集上，結果是單一區域。半徑先乘 2 再四捨五入，偶數再加 1，和 Python 的 `disk` 一樣。`Boundary` 的 `inner` 是原區域減去 3×3 侵蝕，`outer` 是 3×3 膨脹再減去原區域。

範例是 `samples/export_circle.viz`：圓形開運算清掉細雜點，再膨脹，並取出外邊界。

更早完成的還有定義域、裁切、塗色、依距離連通、矩形形態學、補洞、篩選、個數和重心。有 `.pyd` 的形狀匹配路徑沒有改。

## 接著做

1. **產生圓形區域。** `gen_circle` 是實心圓，和形態學的橢圓核不是同一組像素。匯出時要和 `cv2.circle` 的填滿結果相同，包含貼在影像邊上的裁切。
2. **標定仍只有一台相機，也不讀描述檔。** `'5,7,10'` 可用，`board.cpd` 會被拒絕。

Windows 執行檔仍由 `build_windows.bat` 在使用者的電腦上產出。這台不是 Windows，沒有產生 exe。
