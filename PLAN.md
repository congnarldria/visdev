# VisDev 1.1

C# 形狀比對已經先在高斯金字塔的粗圖搜尋，再回到原圖。原圖用分數曲面的拋物線修列、行和角度，再沿邊緣法線做最小二乘。`samples/shapes_scene.png` 的 4 個和 Python 差在 1.5 像素、1 度以內。平移半個像素的方塊落在列 51.5、行 46.5 的 0.45 像素內。

Windows 執行時仍用 `visdev_match.pyd`。`inspect_locate.viz` 的分數仍高於 0.9，沒有改。棋盤 `'5,7,10'` 的 C# 仍是 5 列、7 行、邊長 10、35 個角點。次像素邊緣仍是行 15.5。找線、找圓的飛點仍拉不走。Visual Studio 開 `dotnet/Vizion.Display.sln`，起始專案 `Vizion.Operators.Check`，按 F5。說明在 `dotnet/README.md`。

最近開過的程式沒有做。孔距不做。

Windows 仍用 `app/engine/visdev_match.pyd`。開相機仍先問 DirectShow，打不開再試 OpenCV。開 COM 仍用原來的 API。單台相機的九個參數不改。Linux 的 `visdev_match.so` 不進版控。

## 下一項

C# 正規相關。用同一張 `shape_template.png` 和 `shapes_scene.png`，教導框仍是 15 到 164。先在金字塔粗圖做正規相關，再在原圖把列、行修到次像素。找到 4 個，位置對上 Python 的那 4 個姿態。不改 `find_ncc_model` 的結果，也不改 `.pyd`。

先不要做這些：最近開過的程式、孔和孔的間距、六角形標定板的尋找標記對位、更多 Modbus 功能碼、不合格圖的上一張下一張、編輯器局部更新、公差列再改版。`'5,7,10'` 和矩形圓點板維持可用。

## 維持不變

產線白塊合格 3、不合格 2，良率 60%。孔數產線合格 2、不合格 3，良率 40%。孔數 4 孔、直徑約 28、偏差 0。字元 VISDEV 168。色彩面積 1200、色調 60。
