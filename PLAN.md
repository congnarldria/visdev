# VisDev 1.1

次像素邊緣已經能對照。`samples/inspect_edge.viz` 左半是 0、右半是 255，邊緣在行 15.5，從列 0 到列 23。C# 的 `Operators.EdgesSubPix` 用同一張圖，得到同一個位置。差幾個灰階的雜訊不會留下。

找線、找圓仍在 `samples/inspect_fit.viz`。C# 的 `FitLine`、`FitCircle` 用同一組點。Visual Studio 開 `dotnet/Vizion.Display.sln`，起始專案 `Vizion.Operators.Check`，按 F5。說明在 `dotnet/README.md`。

最近開過的程式沒有做。孔距不做。

Windows 仍用 `app/engine/visdev_match.pyd`。開相機仍先問 DirectShow，打不開再試 OpenCV。開 COM 仍用原來的 API。單台相機的九個參數不改。Linux 的 `visdev_match.so` 不進版控。

## 下一項

形狀比對的 C# 參考。Windows 上的匹配仍用 `app/engine/visdev_match.pyd`，數字不改。C# 先做一份可讀的呼叫範例或純 C# 的簡化比對，用同一張樣板和場景，說清楚和 `.pyd` 哪裡相同、哪裡不同。標定的九個參數這輪不改。

先不要做這些：最近開過的程式、孔和孔的間距、六角形標定板的尋找標記對位、更多 Modbus 功能碼、不合格圖的上一張下一張、編輯器局部更新、公差列再改版。

## 維持不變

產線白塊合格 3、不合格 2，良率 60%。孔數產線合格 2、不合格 3，良率 40%。孔數 4 孔、直徑約 28、偏差 0。字元 VISDEV 168。色彩面積 1200、色調 60。
