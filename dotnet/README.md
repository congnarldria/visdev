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

## 形狀比對

`Operators.FindShape` 讀 `samples/shape_template.png` 和 `samples/shapes_scene.png`，教導框是列 15 到 164、行 15 到 164，和 `samples/inspect_locate.viz` 一樣。場景裡找到 4 個，位置和 Python 差在 12 像素以內。

相同的地方：同一張樣板、同一張場景、同一個教導框、原點是教導框中心、找到 4 個。

不同的地方：Windows 上 VisDev 執行時仍用 `app/engine/visdev_match.pyd`，那裡的分數都高於 0.9。這份 C# 沒有金字塔，也沒有 `least_squares`。分數是邊緣方向有多一致，不是 `.pyd` 的分數。角度先每 4 度掃一圈，再在候選附近細化。

```csharp
var hits = Operators.FindShape(scene, template, 15, 15, 164, 164);
```

自我檢查會從目前目錄往上找 `samples/`。Visual Studio 按 F5 時，工作目錄若在 `bin` 裡也找得到。

## 棋盤標定板

`CalibPlate.Parse("5,7,10")` 是內角點 5 列、7 行、方格邊長 10，角點共 35 個。`5x7x10` 和 `5，7，10` 一樣。物點是 X = 行索引 × 10、Y = 列索引 × 10、Z = 0。第一點是 (0,0,0)，最後一點是 (60,40,0)。OpenCV 的 `patternSize` 是寬 7、高 5。

這份不估計相機的九個參數，也不讀 `.cpd`。圓點板和六角形標定板仍由 Python 處理。單台相機標完的九個參數沒有改。

```csharp
var plate = CalibPlate.Parse("5,7,10");
int corners = plate.CornerCount;
var last = plate.Point(4, 6);
```

既有的 `Threshold`、`Connection`、`AreaCenterOf` 仍在 `Operators`。讀圖目前讀 PNG。
