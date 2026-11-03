# VisDev 1.1

矩形圓點板的 C# 參考已經能讀描述文字。`CalibPlate.ParseDots` 讀 `samples/calplate_4x5.cpd`：4 列、5 行、20 個標記、間距 0.01 公尺、亮標記。第一點是 (0, 0.03, 0)，最後一點是 (0.04, 0, 0)，和 Python 一樣按 Y 由大到小、X 由小到大排。尋找標記那一行略過，不做對位。不估計相機的九個參數。棋盤 `'5,7,10'` 仍是 5 列、7 行、邊長 10、35 個角點，物點從 (0,0,0) 到 (60,40,0)。`Parse` 仍不把路徑當成棋盤。

C# 正規相關先用 Keys 三次內插修列、行和角度。平坦的峰留在取樣點上。接著沿灰階梯度做 Lucas-Kanade，只修列和行。分數沒有變好就不留下。`shapes_scene.png` 的 4 個姿態仍和 Python 差在 1 像素、1.5 度以內。`ignoreGlobalPolarity` 為真時黑白對調仍算對上。不改 `find_ncc_model`，也不改 `.pyd`。

形狀比對的 C# 仍先在粗圖搜尋，再沿邊緣法線做最小二乘。那 4 個和 Python 差在 1.5 像素、1 度以內。`inspect_locate.viz` 的分數仍高於 0.9。次像素邊緣仍是行 15.5。找線、找圓的飛點仍拉不走。Visual Studio 開 `dotnet/Vizion.Display.sln`，起始專案 `Vizion.Operators.Check`，按 F5。說明在 `dotnet/README.md`。

最近開過的程式沒有做。孔距不做。

Windows 仍用 `app/engine/visdev_match.pyd`。開相機仍先問 DirectShow，打不開再試 OpenCV。開 COM 仍用原來的 API。單台相機的九個參數不改。Linux 的 `visdev_match.so` 不進版控。

## 下一項

C# 形狀比對的列、行、角度內插也改用同一套 Keys 三次。平坦的峰留在取樣點上。只在分數沒有變差時才取代現在的拋物線，沿邊緣法線的最小二乘仍接在後面。不改 `visdev_match.pyd`，`inspect_locate.viz` 的分數仍要高於 0.9。

先不要做這些：最近開過的程式、孔和孔的間距、六角形標定板的尋找標記對位、更多 Modbus 功能碼、不合格圖的上一張下一張、編輯器局部更新、公差列再改版。`'5,7,10'` 和矩形圓點板維持可用。

## 維持不變

產線白塊合格 3、不合格 2，良率 60%。孔數產線合格 2、不合格 3，良率 40%。孔數 4 孔、直徑約 28、偏差 0。字元 VISDEV 168。色彩面積 1200、色調 60。
