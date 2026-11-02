# VisDev

VisDev 是一套桌面視覺開發環境。程式由 HALCON 風格的呼叫組成，算子名稱前面加 `Vizion.`：

```
Vizion.read_image (Image, 'samples/parts.png')
Vizion.threshold (Image, Region, 60, 255)
Vizion.connection (Region, Parts)
```

指定用 `:=`。註解以 `*` 開頭。範例在 `samples/`，副檔名是 `.hdv` 和 `.viz`。

## 執行

在 Windows 上，`run.bat` 會用本機虛擬環境啟動 `app/desktop.py`，並打開編輯視窗。

裝好 `requirements.txt` 之後，也可以從殼層啟動：

```
python -m app.desktop
```

不要桌面視窗、只開伺服器：

```
uvicorn app.main:app --host 127.0.0.1 --port 8765
```

瀏覽器打開 `http://127.0.0.1:8765/`。存檔放在 `workspace/`。

COM 埠和 DirectShow 相機使用 Windows API。在其他系統上，這些呼叫會像裝置不存在一樣失敗。`loopback` 序列埠、檔案、socket，以及其他視覺算子仍然可以跑。形狀和 NCC 匹配若找得到 `visdev_match.pyd` 就用它，否則用 Python 實作。

## 測試

```
python -m pytest tests/test_engine.py
node tests/test_view_key.js
```

預設本機 OCR 的測試需要已安裝的 RapidOCR。其餘測試不用。
