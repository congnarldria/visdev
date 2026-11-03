# VisDev 1.1

形狀比對的 C# 參考已經能用。`Operators.FindShape` 讀 `samples/shape_template.png` 和 `samples/shapes_scene.png`，教導框是 15 到 164，找到 4 個，位置和 Python 差在 12 像素以內。Windows 執行時仍用 `app/engine/visdev_match.pyd`，`samples/inspect_locate.viz` 的分數仍高於 0.9。C# 沒有金字塔，也沒有 `least_squares`，分數不是 `.pyd` 的分數。

次像素邊緣仍是行 15.5。找線、找圓的飛點仍拉不走。Visual Studio 開 `dotnet/Vizion.Display.sln`，起始專案 `Vizion.Operators.Check`，按 F5。說明在 `dotnet/README.md`。

最近開過的程式沒有做。孔距不做。

Windows 仍用 `app/engine/visdev_match.pyd`。開相機仍先問 DirectShow，打不開再試 OpenCV。開 COM 仍用原來的 API。單台相機的九個參數不改。Linux 的 `visdev_match.so` 不進版控。

## 下一項

標定的 C# 參考。只示範棋盤 `'5,7,10'` 怎麼解釋成內角點和方格邊長，並對一下 Python 已有的角點數量。不改單台相機標完的九個參數，也不做六角形標定板。

先不要做這些：最近開過的程式、孔和孔的間距、六角形標定板的尋找標記對位、更多 Modbus 功能碼、不合格圖的上一張下一張、編輯器局部更新、公差列再改版。

## 維持不變

產線白塊合格 3、不合格 2，良率 60%。孔數產線合格 2、不合格 3，良率 40%。孔數 4 孔、直徑約 28、偏差 0。字元 VISDEV 168。色彩面積 1200、色調 60。
