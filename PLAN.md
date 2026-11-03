# VisDev 1.1

棋盤 `'5,7,10'` 的 C# 參考已經能用。`CalibPlate.Parse` 得到內角點 5 列、7 行、方格邊長 10，角點 35 個。物點從 (0,0,0) 到 (60,40,0)，和 Python 的 `_object_points` 相同。OpenCV 的 patternSize 是寬 7、高 5。不估計九個參數，也不讀 `.cpd` 或六角形標定板。

形狀比對的 C# 仍找到 4 個，位置和 Python 差在 12 像素以內。Windows 執行時仍用 `visdev_match.pyd`。次像素邊緣仍是行 15.5。找線、找圓的飛點仍拉不走。Visual Studio 開 `dotnet/Vizion.Display.sln`，起始專案 `Vizion.Operators.Check`，按 F5。說明在 `dotnet/README.md`。

最近開過的程式沒有做。孔距不做。

Windows 仍用 `app/engine/visdev_match.pyd`。開相機仍先問 DirectShow，打不開再試 OpenCV。開 COM 仍用原來的 API。單台相機的九個參數不改。Linux 的 `visdev_match.so` 不進版控。

## 下一項

C# 形狀比對加上金字塔和次像素精修。現在是每 4 度全圖掃描，分數不是 `.pyd` 的分數。先在粗圖找候選，再在原圖沿邊緣修列、行和角度。Windows 的 `.pyd` 和 `inspect_locate.viz` 的分數仍高於 0.9，不改。

先不要做這些：最近開過的程式、孔和孔的間距、六角形標定板的尋找標記對位、更多 Modbus 功能碼、不合格圖的上一張下一張、編輯器局部更新、公差列再改版。`'5,7,10'` 和矩形圓點板維持可用。

## 維持不變

產線白塊合格 3、不合格 2，良率 60%。孔數產線合格 2、不合格 3，良率 40%。孔數 4 孔、直徑約 28、偏差 0。字元 VISDEV 168。色彩面積 1200、色調 60。
