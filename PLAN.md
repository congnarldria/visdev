# VisDev 1.1

產線的不合格圖現在帶著標註。`samples/line_batch` 仍是合格 3、不合格 2。沒過面積的小塊畫成紅色，圖上寫 NG，存進 `workspace/line_ng`。按「看不合格圖」看到的就是這張，不是原圖。

圖形視窗的「存畫面」仍另寫 `workspace/view.png`。孔的「設為預期」仍把這次的圓心寫回 `WantRow`、`WantCol`。

Windows 仍用 `app/engine/visdev_match.pyd`。開相機仍先問 DirectShow，打不開再試 OpenCV。開 COM 仍用原來的 API。Linux 的 `visdev_match.so` 不進版控。

## 下一項

產線資料夾也可以檢查孔，不只看白塊面積。用同一套圓心、個數和直徑規則，把每張的判定和不合格圖留下來。`samples/holes.png` 的 4 個孔、直徑約 28、偏差 0 仍是單張範本的基準。

先不要做這些：六角形標定板的尋找標記對位、更多 Modbus 功能碼、不合格圖的上一張下一張、編輯器局部更新再細修、公差列再改版。`'5,7,10'` 和矩形圓點板維持可用。單台相機的九個參數不改。

## 維持不變

產線輸出可在圖形視窗選 TCP、序列埠或 Modbus。預設是 TCP loopback，合格 3、不合格 2。色彩範本的綠塊面積 1200、色調 60。字元範本讀出 VISDEV 168。缺件範本應有 8 個零件。
