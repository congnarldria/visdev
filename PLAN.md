# VisDev 1.1

這一版在這裡停下。孔的圓心沒有做，孔數檢測維持可用：`samples/holes.png` 有 4 個暗孔，少了就不合格。

Windows 仍用 `app/engine/visdev_match.pyd`。開相機仍先問 DirectShow，打不開再試 OpenCV。開 COM 仍用原來的 API。Linux 的 `visdev_match.so` 不進版控。

## 這一版有的

產線輸出可在圖形視窗選 TCP、序列埠或 Modbus。預設是 TCP loopback，合格 3、不合格 2。色彩範本的綠塊面積 1200、色調 60。字元範本讀出 VISDEV 168。缺件範本應有 8 個零件。

## 沒有做

孔的圓心偏差。六角形標定板的尋找標記對位。`'5,7,10'` 和矩形圓點板可以用。
