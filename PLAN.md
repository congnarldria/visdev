# VisDev 1.1

孔數產線已經能用。`samples/hole_batch` 有 5 張，規則和單張孔數相同：4 個暗孔，圓心在列 50 或 120、行 50 或 170，偏差不超過 2，直徑 26 到 30。合格 2、不合格 3。少一孔、圓心偏 10、直徑約 40 各一張。不合格圖在 `workspace/line_ng`，十字是紅色並寫 NG。明細在 `workspace/line_holes.csv`。

白塊產線仍是 `samples/line_batch`，合格 3、不合格 2。單張 `samples/holes.png` 仍是 4 孔、直徑約 28、偏差 0。

Windows 仍用 `app/engine/visdev_match.pyd`。開相機仍先問 DirectShow，打不開再試 OpenCV。開 COM 仍用原來的 API。Linux 的 `visdev_match.so` 不進版控。

## 下一項

量孔和孔之間的間距。現在只看每個圓心離預期位置有多遠，看不出兩孔被拉開或擠在一起。在孔數範本算出相鄰孔的距離，超出範圍就不合格。

先不要做這些：六角形標定板的尋找標記對位、更多 Modbus 功能碼、不合格圖的上一張下一張、編輯器局部更新再細修、公差列再改版。`'5,7,10'` 和矩形圓點板維持可用。單台相機的九個參數不改。

## 維持不變

產線輸出可在圖形視窗選 TCP、序列埠或 Modbus。預設是 TCP loopback，合格 3、不合格 2。色彩範本的綠塊面積 1200、色調 60。字元範本讀出 VISDEV 168。缺件範本應有 8 個零件。
