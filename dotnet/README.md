# Vizion C# 參考

這份是演算法的 C# 參考，不是把 Python 桌面程式打包成 exe。Windows 上的 VisDev 仍用原來的開相機順序和 `visdev_match.pyd`。

## 用 Visual Studio 打開

1. 安裝 Visual Studio 2022，工作負載選「.NET 桌面開發」。要建 .NET Framework 4.8 時，個別元件再勾「.NET Framework 4.8 目標套件」。
2. 檔案、開啟、專案/方案，選 `dotnet/Vizion.Display.sln`。
3. 方案總管裡，對 `Vizion.Operators.Check` 按右鍵，設為起始專案。
4. 上方目標選 `net8.0`，按 F5。主控台印出「算子自我檢查通過」就表示讀圖、二值化、連通、重心、找線、找圓都過了。

要看一條完整的檢查，把起始專案改成 `Vizion.Inspect.Sample`，同樣按 F5。它會在輸出目錄寫 `model.png` 和 `part.png`，再讀回來：去掉亮點、形狀對上轉了 90 度的零件、轉正、量寬度 70 像素（平面上 140）、孔的圓度 0.95，最後印出「判定  合格」。

命令列也可以：

```
dotnet run --project dotnet/Vizion.Operators.Check/Vizion.Operators.Check.csproj -f net8.0
dotnet run --project dotnet/Vizion.Inspect.Sample/Vizion.Inspect.Sample.csproj -f net8.0
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

`Operators.FindShape` 讀 `samples/shape_template.png` 和 `samples/shapes_scene.png`，教導框是列 15 到 164、行 15 到 164，和 `samples/inspect_locate.viz` 一樣。場景裡找到 4 個。位置和 Python 差在 1.5 像素以內，角度差在 1 度以內。

相同的地方：同一張樣板、同一張場景、同一個教導框、原點是教導框中心、找到 4 個。先在高斯金字塔的粗圖搜尋，再回到原圖。原圖上用分數曲面的拋物線修列、行和角度，再沿邊緣法線做最小二乘。

不同的地方：Windows 上 VisDev 執行時仍用 `app/engine/visdev_match.pyd`，那裡的分數都高於 0.9。這份 C# 的分數是邊緣方向有多一致，不是 `.pyd` 的分數。粗層角度步距是 4 度。

```csharp
var hits = Operators.FindShape(scene, template, 15, 15, 164, 164);
```

## 正規相關

`Operators.FindNcc` 用同一張樣板、同一張場景、同一個教導框。先在高斯金字塔的粗圖做灰階相關，再回到原圖。分數用 Keys 三次內插（a = -0.5）修列、行和角度。平坦的峰留在取樣點上，不跟著很淺的斜坡滑。4 個姿態和 Python 差在 1 像素、1.5 度以內。

內插之後沿灰階梯度做 Lucas-Kanade，只修列和行，角度維持內插的結果。分數沒有變好就不留下。

這不是 `find_ncc_model`。Windows 執行時的正規相關仍走原來的路徑，分數沒有改。`ignoreGlobalPolarity: true` 會把相關係數取絕對值，黑白對調仍算對上。沒開時，對調不算對上。

```csharp
var hits = Operators.FindNcc(scene, template, 15, 15, 164, 164);
```

自我檢查會從目前目錄往上找 `samples/`。Visual Studio 按 F5 時，工作目錄若在 `bin` 裡也找得到。

## 棋盤標定板

`CalibPlate.Parse("5,7,10")` 是內角點 5 列、7 行、方格邊長 10，角點共 35 個。`5x7x10` 和 `5，7，10` 一樣。物點是 X = 行索引 × 10、Y = 列索引 × 10、Z = 0。第一點是 (0,0,0)，最後一點是 (60,40,0)。OpenCV 的 `patternSize` 是寬 7、高 5。

`CalibPlate.ParseDots` 讀矩形圓點板的描述文字，和 `samples/calplate_4x5.cpd` 一樣：4 列 5 行、20 個標記、間距 0.01 公尺、亮標記。第一點是 (0, 0.03, 0)，最後一點是 (0.04, 0, 0)。尋找標記那一行會略過，不做對位。路徑仍不會被 `Parse` 當成棋盤。不估計相機的九個參數，也不做六角形標定板。單台相機標完的九個參數沒有改。

```csharp
var plate = CalibPlate.Parse("5,7,10");
int corners = plate.CornerCount;
var last = plate.Point(4, 6);
```

## 區域特徵和寬度

`Operators.ShapeOf` 給圓度、矩形度、方向，還有等效橢圓的長短軸。橫條方向是 0，直條是 1.5708。圓度是 4π 乘面積除以周長平方。矩形度是面積除以輪廓最小外接矩形，實心橫條大約 1.32。

`Operators.MeasureWidth` 量一列亮條的左右邊緣。左半 0、中間 255、右半 0 時，邊緣在 15.5 和 47.5，寬度是 32。兩條已經擬合的邊緣用 `Operators.LineDistance`，垂直距離同樣是 32。

## 從棋盤圖到內參

`ChessboardPhoto.Render` 用已知內參和姿態把 `'5,7,10'` 畫成灰階圖。`ChessboardPhoto.FindCorners` 從黑格的四邊交點取出 35 個內角點，順序和 `CalibPlate.Point` 相同。再送進 `PlanarCalib.Estimate`。合成的 600、610、320、240，焦距差在 8 以內，主點差在 2 以內。不改 Python 從真實標定圖算出的九個參數，也不找六角形標記。

## 相機內參

`PlanarCalib.Estimate` 用至少三個視角的平面對應點估計 fx、fy、cx、cy。沒有畸變時，合成的 600、610、320、240 會回到原值。`EstimateDistorted` 再估計徑向 k1，模型是 x' = x (1 + k1 r²)。不從照片裡找角點，也不改 Python `calibrate_cameras` 已經算出的九個參數。

```csharp
var shape = Operators.ShapeOf(region);
var width = Operators.MeasureWidth(image, row: 16);
var camera = PlanarCalib.Estimate(objectX, objectY, views);
```

`GrayOpenRectangle` 和 `GrayCloseRectangle` 是灰階矩形開運算、閉運算。3×3 遮罩會去掉單獨的 200，留下 3×3 的 180；周圍都是 180 時，中間的 0 會補成 180。`GrayOpenCircle` 的半徑 1 是十字。3×3 方塊開完剩下十字，單獨的亮點消失。`GrayCloseCircle` 用同一個十字把單獨的暗點補成周圍的灰階。9×9 圖中間和角落的 0 會變成 180。

`Undistort` 用 k1 把圖校成直線模型。行 36 的理想位置會到原圖的 34.72 取樣。`ImageToPlane` 把像素投到 Z=0 的平面。沒有畸變時 (列 20, 行 30) 是平面 x 50；k1 為 -0.5 時約 51.730。`RotateNearest` 正角度在畫面上順時針。橫條轉 90 度後，寬度仍是 8 像素，在焦距 40、距離 200 時是 40 個世界單位。`MeasurePlaneWidth` 先去掉 k1，再把左右邊緣投到平面上，同一條亮條約 40。

既有的 `Threshold`、`Connection`、`AreaCenterOf` 仍在 `Operators`。矩形和圓形的膨脹、侵蝕、開運算、閉運算也還在。讀圖目前讀 PNG。
