# VisDev 1.1 beta

這是 beta，不是正式版。完成度夠了才會拿掉 beta。

VisDev 是一套桌面視覺開發環境。程式寫法接近 HDevelop，算子名稱前面加 `Vizion.`，影像計算用 OpenCV。

```
Vizion.read_image (Image, 'samples/parts.png')
MinGray := 60
Vizion.threshold (Image, Region, MinGray, 255)
Vizion.connection (Region, Parts, 1)
```

指定用 `:=`。註解以 `*` 開頭。`T[1]` 從 1 開始。範例在 `samples/`，副檔名是 `.hdv` 和 `.viz`。

## 怎麼開始

在 Windows 上，`run.bat` 會用本機虛擬環境啟動，並打開編輯視窗。視窗標題是 VisDev 1.1 beta。

裝好 `requirements.txt` 之後，也可以：

```
python -m app.desktop
```

不要桌面視窗、只開伺服器：

```
uvicorn app.main:app --host 127.0.0.1 --port 8765
```

瀏覽器打開 `http://127.0.0.1:8765/`。

檔案選單：

- **範本**打開就能跑。零件檢測讀 `samples/parts.png`，這張圖是 8 個零件。瑕疵讀 `samples/hard_contrast.png`：暗塊 1 個，面積 3550；亮點 1 個，面積 144。邊緣量測和卡尺讀 `samples/gauge.png`。卡尺的矩形寬 140、高 90，圓的直徑約 83.93。條紋寬度讀 `samples/stripes.png`，最寬的亮條是 22。形狀定位在場景裡找到 4 個。相對姿態用寫入的角點示範兩台相機，第 k 組是同一次擺放，`Pose1` 的第一個數約 0.04 公尺。條碼讀 `samples/codes.png`，`Text` 是 4006381333931，`QrText` 是 VISDEV-3。點量規或直線量規時，可以按「線」在圖上拖，起點到終點的方向會保留。圖上已有影像時，確認前會畫出邊緣和首尾間距。
- **新增、開啟、儲存**。存檔在 `workspace/`。**專案**可以把多支程式放在同一個資料夾，也可以改名、打開資料夾。存進專案時，程式讀的 `workspace/` 圖會拷進去，路徑改成檔名。`samples/` 的圖仍共用。從檔案總管貼上的反斜線路徑也能讀。下次打開會回到上次的專案。
- **載入影像**把圖放進 `workspace/`，路徑可寫進 `read_image`。
- **匯出 C#** 涵蓋讀圖、定義域、二值化、連通、篩選、矩形和圓形形態學、個數和重心。

執行是 F5，單步是 F6，步進程序是 F7。點算子名稱可改參數。有座標的算子可在圖上框選。說明選單裡的「使用說明」有同樣的步驟，不用另開這個檔。

要在 Windows 上打包，執行 `build_windows.bat`。需要 Python 3.9 或更新版本、`app/engine/visdev_match.pyd`，以及 PyInstaller。腳本會分別說明缺哪一個。執行檔在使用者的電腦上產出，是 `dist\VisDev\VisDev.exe`。

## 標定

`set_calib_data_calib_object` 接受棋盤三個數，例如 `'5,7,10'`（內角點 5 列、7 行、方格邊長 10），也接受 `.cpd` 或 `.descr`。圓點板用 `findCirclesGrid`，亮標記會先反相。範例描述檔是 `samples/calplate_4x5.cpd`。

`create_calib_data` 的相機數可以大於 1。每台相機用自己的影像估計九個參數：fx、fy、cx、cy、k1、k2、p1、p2、k3。標完之後，`get_calib_data` 的 `camera/pose` 是這台相機相對第 0 台的姿態，七個數是 Tx、Ty、Tz、Rx、Ry、Rz、0。第 0 台是 0。每一台的第 k 組必須是同時看到的同一塊板。

標定結束時，圖形視窗會畫出找到的點和誤差。多張由上到下排在一起，點某一張可看原圖，再點一下回到全部。偏差特別大的點是紅色。

## 已知限制

- 六角形標定板的尋找標記還沒有對位。`'5,7,10'` 和矩形圓點板可以用。
- 相對姿態只在各相機的第 k 組是同一次擺放時才有意義。張數對不上的相機不會得到姿態。
- 標定物這版只能一塊。
- 形狀匹配在 Windows 使用 `app/engine/visdev_match.pyd`。Linux 的 `visdev_match.so` 不進版控，要在本機編譯。
- COM 埠和 DirectShow 相機使用 Windows API。其他系統上這些呼叫會像裝置不存在一樣失敗。檔案相機、`loopback` 序列埠和其餘視覺算子仍然可以跑。
- 本機 OCR 需要已安裝的 RapidOCR。沒有安裝時，自動模式不會用到它。
- C# 專案有 `net48` 與 `net8.0`。這份環境只保證 `net8.0` 能建置。

## 測試

```
python -m pytest tests/test_engine.py
node tests/test_view_key.js
node tests/test_var_list.js
node tests/test_highlight_sync.js
```
