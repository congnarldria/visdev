# VisDev 1.1

VisDev 是一套桌面視覺開發環境。程式寫法接近 HDevelop，算子名稱前面加 `Vizion.`，影像計算用 OpenCV。

```
Vizion.read_image (Image, 'samples/parts.png')
MinGray := 60
Vizion.threshold (Image, Region, MinGray, 255)
Vizion.connection (Region, Parts, 1)
```

指定用 `:=`。註解以 `*` 開頭。`T[1]` 從 1 開始。範例在 `samples/`，副檔名是 `.hdv` 和 `.viz`。

## 怎麼開始

在 Windows 上，`run.bat` 會用本機虛擬環境啟動，並打開編輯視窗。視窗標題是 VisDev 1.1。

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
- **匯出 C#** 涵蓋讀圖、定義域、二值化、連通、篩選、矩形和圓形形態學、個數和重心。可建置的參考在 `dotnet/`，找線、找圓和次像素邊緣的 C# 與對應範例同一組數字。Visual Studio 的開法寫在 `dotnet/README.md`。

執行是 F5，單步是 F6，步進程序是 F7。跑完後，狀態列和圖形視窗會寫出直徑、寬度、間距、條碼、個數、合格或不合格。卡尺、條紋和瑕疵的上下限顯示在圖形視窗，改完再按執行即可。瑕疵這裡改的是灰階範圍。改過的範圍會記在這台電腦，再打開同一支範本會填回去。按「預設」回到程式裡的數字。產線範本讀 `samples/line_batch`，合格 3 張、不合格 2 張。不合格的圖在 `workspace/line_ng`，畫著沒過的區域和 NG 字。明細在 `workspace/line_result.csv`。這支用檔案資料夾，不開相機。產線相機用編號 0，Windows 仍先問 DirectShow，打不開再試 OpenCV。沒有相機時，請改跑檔案產線，用 `samples/line_batch`。按停止就停，已經計的張數留著。執行中狀態列也會寫出合格和不合格。明細在 `workspace/line_camera.csv`，每一列是這張的 OK 或 NG。出錯會關掉相機和 CSV。再跑一次會先清掉上一輪的不合格圖。`LastNg` 是最近一張不合格圖的檔名，全部合格時是空的。按「看不合格圖」可把它開回圖形視窗，再按一次回到程式畫面。按「看明細」可看 CSV，也可以只看不合格。按「統計」可看整份明細的總數、合格、不合格和良率。檔案選單的「範本一覽」列出每一支該有的結果，點一列就打開。點不合格那一列會打開那張圖。產線輸出把每一張的判定送成一行 `VISDEV,OK 或 NG,合格張數,不合格張數,檔名`。範本用 TCP loopback，`Wire` 讀回 5 行，合格仍是 3、不合格仍是 2。改成主機埠號或 COM 名稱就送到外面。Windows 開 COM 的方式沒有改。色彩範本讀 `samples/color_part.png`，綠塊面積 1200、色調 60，紅塊不算。色調超出 50 到 70，或面積超出 1100 到 1300，判定就是不合格。字元範本讀 `samples/ocr_label.png`。`Chars` 是讀到的字，預期字預設 VISDEV 168，圖形視窗可以改，再按執行。改過的字會記在這台電腦。讀錯就不合格。外框畫在圖上。讀到的字可以另送一行 `VISDEV,TEXT,文字`，不改 OK、NG 那一行。Modbus TCP 一次可讀或寫最多 16 個連續暫存器。合格仍寫成 1、不合格寫成 2。不開 COM。產線輸出在圖形視窗先選 TCP、序列埠或 Modbus。預設是 TCP loopback，合格仍是 3、不合格仍是 2。缺件範本讀 `samples/parts.png`，應有 8 個。孔數範本讀 `samples/holes.png`，4 個暗孔，每個面積 613，直徑約 28。圓心在列 50 或 120、行 50 或 170，偏差是 0。超過 2，或直徑不在 26 到 30，就不合格，十字改成紅色。按「設為預期」把這次量到的圓心寫進 `WantRow`、`WantCol`，再按執行，偏差就是 0。跑完後按「存畫面」，底圖、區域和十字寫進 `workspace/view.png`，讀圖路徑不會改。孔數產線讀 `samples/hole_batch`，同一套規則，合格 2、不合格 3。白塊產線仍是合格 3、不合格 2。點算子名稱可改參數。有座標的算子可在圖上框選。說明選單裡的「使用說明」有每支範本和對應的數字。

要在 Windows 上打包，執行 `build_windows.bat`。需要 Python 3.9 或更新版本、`app/engine/visdev_match.pyd`，以及 PyInstaller。腳本會分別說明缺哪一個。執行檔在使用者的電腦上產出，是 `dist\VisDev\VisDev.exe`。

## 標定

`set_calib_data_calib_object` 接受棋盤三個數，例如 `'5,7,10'`（內角點 5 列、7 行、方格邊長 10），也接受 `.cpd` 或 `.descr`。圓點板用 `findCirclesGrid`，亮標記會先反相。範例描述檔是 `samples/calplate_4x5.cpd`。

`create_calib_data` 的相機數可以大於 1。每台相機用自己的影像估計九個參數：fx、fy、cx、cy、k1、k2、p1、p2、k3。標完之後，`get_calib_data` 的 `camera/pose` 是這台相機相對第 0 台的姿態，七個數是 Tx、Ty、Tz、Rx、Ry、Rz、0。第 0 台是 0。每一台的第 k 組必須是同時看到的同一塊板。

標定結束時，圖形視窗會畫出找到的點和誤差。多張由上到下排在一起，點某一張可看原圖，再點一下回到全部。偏差特別大的點是紅色。

## 已知限制

- 孔的預期位置在 `WantRow`、`WantCol`。圖形視窗的「設為預期」把這次量到的圓心寫回去。偏差仍是整組裡最大的那一個距離。
- 產線不合格圖由程式存成帶區域和 NG 字的圖。圖形視窗的「存畫面」另寫 `workspace/view.png`。
- 六角形標定板的尋找標記還沒有對位。`'5,7,10'` 和矩形圓點板可以用。
- 相對姿態只在各相機的第 k 組是同一次擺放時才有意義。張數對不上的相機不會得到姿態。
- 標定物這版只能一塊。
- 形狀匹配在 Windows 使用 `app/engine/visdev_match.pyd`。Linux 的 `visdev_match.so` 不進版控，要在本機編譯。
- COM 埠和 DirectShow 相機使用 Windows API。其他系統上這些呼叫會像裝置不存在一樣失敗。檔案相機、`loopback` 序列埠和其餘視覺算子仍然可以跑。
- 本機 OCR 需要已安裝的 RapidOCR。沒有安裝時，自動模式不會用到它。
- C# 專案有 `net48` 與 `net8.0`。這份環境只保證 `net8.0` 能建置。找線、找圓的呼叫方式見 `dotnet/README.md`。

## 測試

```
python -m pytest tests/test_engine.py
node tests/test_view_key.js
node tests/test_var_list.js
node tests/test_highlight_sync.js
```
