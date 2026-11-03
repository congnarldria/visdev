# VisDev 1.1

找線、找圓已經能對照。`samples/inspect_fit.viz` 打開就能跑：直線飛點拉不走，兩端是列 0、行 0 和列 0、行 30。圓的飛點拉不走，圓心列 50、行 80，半徑 20。C# 的 `Operators.FitLine`、`Operators.FitCircle` 用同一組點。Visual Studio 開 `dotnet/Vizion.Display.sln`，把 `Vizion.Operators.Check` 設成起始專案，按 F5。說明在 `dotnet/README.md`。

最近開過的程式沒有做。孔距不做。

Windows 仍用 `app/engine/visdev_match.pyd`。開相機仍先問 DirectShow，打不開再試 OpenCV。開 COM 仍用原來的 API。單台相機的九個參數不改。Linux 的 `visdev_match.so` 不進版控。

## 下一項

次像素邊緣。Python 的 `edges_sub_pix` 已經有。用同一張階躍圖，讓 C# 量到的邊緣位置和 Python 一致，並加一支打開就能跑的範例。

先不要做這些：最近開過的程式、孔和孔的間距、六角形標定板的尋找標記對位、更多 Modbus 功能碼、不合格圖的上一張下一張、編輯器局部更新、公差列再改版。

## 維持不變

產線白塊合格 3、不合格 2，良率 60%。孔數產線合格 2、不合格 3，良率 40%。孔數 4 孔、直徑約 28、偏差 0。字元 VISDEV 168。色彩面積 1200、色調 60。
