# Vizion C# 參考

這份是演算法的 C# 參考，不是把 Python 桌面程式打包成 exe。Windows 上的 VisDev 仍用原來的開相機順序和 `visdev_match.pyd`。

## 用 Visual Studio 打開

1. 安裝 Visual Studio 2022，工作負載選「.NET 桌面開發」。要建 .NET Framework 4.8 時，個別元件再勾「.NET Framework 4.8 目標套件」。
2. 檔案、開啟、專案/方案，選 `dotnet/Vizion.Display.sln`。
3. 方案總管裡，對 `Vizion.Operators.Check` 按右鍵，設為起始專案。
4. 上方目標選 `net8.0`，按 F5。主控台印出「算子自我檢查通過」就表示讀圖、二值化、連通、重心、找線、找圓都過了。

命令列也可以：

```
dotnet run --project dotnet/Vizion.Operators.Check/Vizion.Operators.Check.csproj -f net8.0
```

`Vizion.Operators` 同時標 `net8.0` 和 `net48`。這台 Linux 環境只保證 `net8.0` 能跑檢查程式。Windows 上若裝了目標套件，`net48` 也能建置。

## 找線、找圓

`Operators.FitLine`、`Operators.FitCircle` 的點是 `(列, 行)`。離模型太遠的飛點會丟掉再擬一次。點都在模型上時，結果和一次擬合一樣。

Python 同一組點用 `samples/inspect_fit.viz`。直線兩端是列 0、行 0 和列 0、行 30。圓心是列 50、行 80，半徑 20。C# 自我檢查用同一組數字。

```csharp
var line = Operators.FitLine(new (double Row, double Col)[]
{
    (0, 0), (0, 10), (8, 15), (0, 20), (0, 30),
});
var circle = Operators.FitCircle(pointsOnTheCircle);
```

## 次像素邊緣

`Operators.EdgesSubPix` 和 `Vizion.edges_sub_pix` 的 canny 相同。寬 32、高 24、左半 0、右半 255 的階躍，邊緣在行 15.5，從列 0 到列 23。Python 範例是 `samples/inspect_edge.viz`。差幾個灰階的雜訊不會留下。

```csharp
var edges = Operators.EdgesSubPix(image, alpha: 1, low: 20, high: 40);
```

既有的 `Threshold`、`Connection`、`AreaCenterOf` 仍在 `Operators`。讀圖目前讀 PNG。
