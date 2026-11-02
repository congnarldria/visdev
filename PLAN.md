# 改進計畫

這一支的起點是本機工作複本的快照。下面是做完快照之後還值得做的事，以及這次已經處理的項目。

## 已完成

1. **圖形視窗不會再因為畫面沒變色就跳過更新。** `app/static/view.js` 的比對改成含影像內容、輪廓點、文字和填色。文字會畫出來。點在封閉輪廓裡面時，選取方式和主視窗一樣。
2. **算子模組在非 Windows 上可以載入。** `serial_ops` 不再在 import 時檢查 `DCB`、載入 `kernel32`。`loopback` 序列埠仍可用，打不開的 COM 仍回「打不開 COMxxx」。`dshow` 改到真正開相機時才載入，失敗會落到 OpenCV。`get_socket_param` 的 `available` 在 Windows 用 `ws2_32`，其他系統用 `FIONREAD`。
3. **重設會關掉作業系統句柄。** `Session.reset` 會關閉檔案、socket、序列埠和相機。工作階段被換掉或超出保留數量時也會關。
4. **檔案算子和 socket 未讀位元組有測試。** 圖形視窗的畫面指紋另有 `tests/test_view_key.js`。
5. **`watersheds` 重新註冊。** 裝飾器原先掛在內部的 `_watershed_markers`，程式和測試都找不到 `watersheds`。
6. **PDF417 和 Aztec 可以建立模型。** `_symbol_kind` 在判斷這兩種碼之前就丟出例外，後面的分支到不了。
7. **測試裡漏掉 `Vizion.` 前綴的呼叫已補上。** 否則解析階段就停住，到不了算子自己的錯誤訊息。
8. **Lepetit 範例的 `for` / `if` 有收尾，並寫出 `BCol`。**
9. **`list_files` 的說明和 `chapters` 的匯出清單。** 說明會提到 `recursive`。`__all__` 含有 `serial_ops`。

## 接著做

10. **三個數值測試仍和 Python 後備實作不一致。** 沒有 `visdev_match.pyd` 時：
    - `ViZion_gauss_image_smoothed_shift_row_is_39.23`：整數峰是 39.5，不是 40；Keys 內插列是 39.28，不是 39.23。
    - `ViZion_gauss_image_least_squares_row_near_39.25`：interpolation 的列同樣是 39.28。
    - `ViZion_create_scaled_shape_model_scale_refined_to_1.07505`：尺度最小二乘從目前的最佳姿態修出去之後分數變差，結果留在 1.0，不是 1.07505。從另一個較正的姿態出發時，同一支函式會修到 1.07505。
11. **`file_ops` 的相對路徑可以用 `..` 離開專案目錄。** 絕對路徑應保持可用。OCR 測試在裝了 `requirements.txt` 裡的 `rapidocr-onnxruntime` 之後可以通過。

## 以後

12. **`wait_seconds` 會佔住處理這次請求的執行緒**，最長 30 秒。
13. **原生匹配器。** `visdev_match.pyd` 只有 Windows 版。其他系統走 `shape_model.py` 和 `ncc_model.py`。把 `match_core.c` 編成那些系統的擴充模組，才能和上面的數值測試對齊。
14. **標成「這版」的算子。** 標定、部分條碼和只能做 byte 的影像，仍會拒絕一部分輸入。一次補一個系列，並附範例程式。
15. **編輯器畫布還沒有自動化測試。** 目前只檢查圖形視窗的畫面指紋。
