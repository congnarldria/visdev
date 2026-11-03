# VisDev 1.1

C# 正規相關已經能用。`Operators.FindNcc` 用 `shape_template.png` 和 `shapes_scene.png`，教導框 15 到 164。先在高斯金字塔的粗圖做灰階相關，再回到原圖。分數平頂取中心，再用拋物線修列、行和角度。4 個姿態和 Python 差在 1 像素、1.5 度以內。`ignoreGlobalPolarity` 為真時，黑白對調仍算對上；沒開則不算。不改 `find_ncc_model`，也不改 `.pyd`。

形狀比對的 C# 仍先在粗圖搜尋，再沿邊緣法線做最小二乘。那 4 個和 Python 差在 1.5 像素、1 度以內。`inspect_locate.viz` 的分數仍高於 0.9。棋盤 `'5,7,10'` 仍是 5 列、7 行、邊長 10、35 個角點。次像素邊緣仍是行 15.5。找線、找圓的飛點仍拉不走。Visual Studio 開 `dotnet/Vizion.Display.sln`，起始專案 `Vizion.Operators.Check`，按 F5。說明在 `dotnet/README.md`。

最近開過的程式沒有做。孔距不做。

Windows 仍用 `app/engine/visdev_match.pyd`。開相機仍先問 DirectShow，打不開再試 OpenCV。開 COM 仍用原來的 API。單台相機的九個參數不改。Linux 的 `visdev_match.so` 不進版控。

## 下一項

C# 正規相關的列、行再沿灰階梯度做一次 Lucas-Kanade，對齊 Python `least_squares` 的那一步。現在的拋物線已經把 4 個姿態收到 1 像素以內。這一步只在分數沒有變差時才留下。不改 `find_ncc_model` 的結果，也不改 `.pyd`。

先不要做這些：最近開過的程式、孔和孔的間距、六角形標定板的尋找標記對位、更多 Modbus 功能碼、不合格圖的上一張下一張、編輯器局部更新、公差列再改版。`'5,7,10'` 和矩形圓點板維持可用。

## 維持不變

產線白塊合格 3、不合格 2，良率 60%。孔數產線合格 2、不合格 3，良率 40%。孔數 4 孔、直徑約 28、偏差 0。字元 VISDEV 168。色彩面積 1200、色調 60。
