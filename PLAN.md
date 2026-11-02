# 改進計畫

C# 的影像已經有定義域。`reduce_domain` 之後，二值化只留下區域裡的像素，圖形視窗裡區域外會變暗。Windows 仍用 `app/engine/visdev_match.pyd`。Linux 的 `visdev_match.so` 不進版控。

## 已完成

`ReduceDomain` 會和既有定義域取交集，`ChangeDomain` 整段換掉，`DomainOf` 在沒有做過限制時是整張圖。`Crop` 會把定義域一起裁走。顯示用 `ForDisplay`，定義域外的像素是原來的三分之一。

同一輪也接上固定灰階的新影像、把區域塗進影像、只在定義域裡塗色，以及改一個像素。範例是 `samples/export_domain.viz`。

更早完成的還有依距離連通、矩形形態學、補洞、篩選、個數和重心。有 `.pyd` 的形狀匹配路徑沒有改。

## 接著做

1. **圓形形態學。** 開運算、閉運算、膨脹、侵蝕現在匯出只做矩形。圓形結構元素是下一個常拿來清雜點的算子。
2. **標定仍只有一台相機，也不讀描述檔。** `'5,7,10'` 可用，`board.cpd` 會被拒絕。

Windows 執行檔仍由 `build_windows.bat` 在使用者的電腦上產出。這台不是 Windows，沒有產生 exe。
