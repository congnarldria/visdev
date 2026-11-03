using System.Text.Json;
using System.Text.Json.Serialization;
using Vizion;

if (args.Length >= 2 && args[0] == "--compare")
{
    SelfTest();
    Compare(args[1]);
    return;
}

SelfTest();
Console.WriteLine("算子自我檢查通過");

static void SelfTest()
{
    var straight = Operators.FitLine(new (double Row, double Col)[]
    {
        (0, 0), (0, 10), (8, 15), (0, 20), (0, 30),
    });
    bool At(double row, double col, double wantRow, double wantCol) =>
        Math.Abs(row - wantRow) < 1e-6 && Math.Abs(col - wantCol) < 1e-6;
    bool lineEnds =
        (At(straight.RowBegin, straight.ColBegin, 0, 0) && At(straight.RowEnd, straight.ColEnd, 0, 30))
        || (At(straight.RowBegin, straight.ColBegin, 0, 30) && At(straight.RowEnd, straight.ColEnd, 0, 0));
    if (!lineEnds)
        throw new InvalidOperationException($"飛點不該把直線拉走，得到 ({straight.RowBegin},{straight.ColBegin})-({straight.RowEnd},{straight.ColEnd})");

    var circle = Operators.FitCircle(new (double Row, double Col)[]
    {
        (50, 100), (60, 97.3205080757), (67.3205080757, 90), (70, 80),
        (67.3205080757, 70), (60, 62.6794919243), (50, 60), (40, 62.6794919243),
        (32.6794919243, 70), (30, 80), (32.6794919243, 90), (40, 97.3205080757),
        (50, 108),
    });
    if (Math.Abs(circle.Row - 50) > 1e-6 || Math.Abs(circle.Column - 80) > 1e-6 || Math.Abs(circle.Radius - 20) > 1e-6)
        throw new InvalidOperationException($"飛點不該把圓拉走，得到列 {circle.Row} 行 {circle.Column} 半徑 {circle.Radius}");

    var step = new byte[24 * 32];
    for (int row = 0; row < 24; row++)
        for (int col = 16; col < 32; col++)
            step[row * 32 + col] = 255;
    var edges = Operators.EdgesSubPix(ImageBuffer.Gray(32, 24, step), 1, 20, 40);
    if (edges.Length != 1 || edges[0].Length != 24)
        throw new InvalidOperationException($"階躍應是 1 條 24 點的邊緣，得到 {edges.Length} 條");
    foreach (var point in edges[0])
    {
        if (Math.Abs(point.Column - 15.5) > 1e-3)
            throw new InvalidOperationException($"邊緣應在行 15.5，得到 {point.Column}");
    }
    if (Math.Abs(edges[0][0].Row) > 1e-3 || Math.Abs(edges[0][^1].Row - 23) > 1e-3)
        throw new InvalidOperationException($"邊緣應從列 0 到列 23，得到 {edges[0][0].Row} 到 {edges[0][^1].Row}");

    var plate = CalibPlate.Parse("5,7,10");
    if (plate.Rows != 5 || plate.Columns != 7 || plate.Square != 10 || plate.CornerCount != 35)
        throw new InvalidOperationException($"5,7,10 應是 5 列 7 行、邊長 10、35 個角點，得到 {plate.Rows}x{plate.Columns} 邊長 {plate.Square}");
    if (plate.PatternWidth != 7 || plate.PatternHeight != 5)
        throw new InvalidOperationException("OpenCV 的 patternSize 應是寬 7、高 5");
    var origin = plate.Point(0, 0);
    var along = plate.Point(0, 1);
    var down = plate.Point(1, 0);
    var last = plate.Point(4, 6);
    if (origin != (0, 0, 0) || along != (10, 0, 0) || down != (0, 10, 0) || last != (60, 40, 0))
        throw new InvalidOperationException($"物點應從 (0,0,0) 到 (60,40,0)，得到 {origin} {along} {down} {last}");
    if (CalibPlate.Parse("5x7x10").CornerCount != 35 || CalibPlate.Parse("5，7，10").CornerCount != 35)
        throw new InvalidOperationException("x 和中文逗號也要能寫成同一塊棋盤");
    try
    {
        CalibPlate.Parse("1,7,10");
        throw new InvalidOperationException("不到 2x2 的棋盤不該被接受");
    }
    catch (ArgumentException)
    {
    }
    try
    {
        CalibPlate.Parse("samples/calplate_4x5.cpd");
        throw new InvalidOperationException("描述檔不該在這份 C# 參考裡被當成棋盤");
    }
    catch (ArgumentException)
    {
    }
    var dotPlate = CalibPlate.ParseDots(File.ReadAllText(SampleFile("calplate_4x5.cpd")));
    if (dotPlate.Rows != 4 || dotPlate.Columns != 5 || dotPlate.MarkCount != 20 || dotPlate.Polarity != "light")
        throw new InvalidOperationException($"圓點板應是 4 列 5 行、20 點、亮標記，得到 {dotPlate.Rows}x{dotPlate.Columns} {dotPlate.MarkCount} {dotPlate.Polarity}");
    if (Math.Abs(dotPlate.Square - 0.01) > 1e-9)
        throw new InvalidOperationException($"圓點間距應是 0.01，得到 {dotPlate.Square}");
    var firstDot = dotPlate.Point(0);
    var lastDot = dotPlate.Point(19);
    if (firstDot != (0, 0.03, 0) || dotPlate.Point(4) != (0.04, 0.03, 0) || dotPlate.Point(5) != (0, 0.02, 0) || lastDot != (0.04, 0, 0))
        throw new InvalidOperationException($"圓點應從 (0,0.03,0) 到 (0.04,0,0)，得到 {firstDot} {dotPlate.Point(4)} {dotPlate.Point(5)} {lastDot}");
    try
    {
        CalibPlate.ParseDots("r 2\nc 2\n0 0 0.1\n");
        throw new InvalidOperationException("標記數和列乘行不合時不該被接受");
    }
    catch (ArgumentException)
    {
    }
    var dark = CalibPlate.ParseDots("r 2\nc 2\np dark\n0 1 0.1\n1 1 0.1\n0 0 0.1\n1 0 0.1\n");
    if (dark.Polarity != "dark" || dark.Point(0) != (0, 1, 0) || Math.Abs(dark.Square - 1) > 1e-9)
        throw new InvalidOperationException($"暗標記應從 (0,1,0) 起、間距 1，得到 {dark.Polarity} {dark.Point(0)} {dark.Square}");

    Region Bar(int row1, int column1, int row2, int column2)
    {
        int count = row2 - row1 + 1;
        var rows = new int[count];
        var columns = new int[count];
        var lengths = new int[count];
        for (int i = 0; i < count; i++)
        {
            rows[i] = row1 + i;
            columns[i] = column1;
            lengths[i] = column2 - column1 + 1;
        }
        return new Region(80, 60, rows, columns, lengths);
    }
    var wideShape = Operators.ShapeOf(Bar(10, 10, 14, 29));
    var tallShape = Operators.ShapeOf(Bar(10, 10, 29, 14));
    if (Math.Abs(wideShape.Orientation) > 1e-9 || Math.Abs(tallShape.Orientation - Math.PI / 2) > 1e-9)
        throw new InvalidOperationException($"橫條方向應是 0、直條應是 1.5708，得到 {wideShape.Orientation} {tallShape.Orientation}");
    if (Math.Abs(wideShape.Circularity - 0.5938738475595072) > 1e-6 || Math.Abs(wideShape.Rectangularity - 1.3157894736842106) > 1e-6)
        throw new InvalidOperationException($"橫條圓度、矩形度不對，得到 {wideShape.Circularity} {wideShape.Rectangularity}");
    if (Math.Abs(wideShape.Ra - 11.532562594670797) > 1e-6)
        throw new InvalidOperationException($"橫條長軸應約 11.53，得到 {wideShape.Ra}");
    var diskRows = new List<int>();
    var diskColumns = new List<int>();
    var diskLengths = new List<int>();
    for (int row = 0; row < 80; row++)
    {
        int left = -1;
        int right = -1;
        for (int column = 0; column < 80; column++)
        {
            int dy = row - 40;
            int dx = column - 40;
            if (dy * dy + dx * dx <= 20 * 20)
            {
                if (left < 0)
                    left = column;
                right = column;
            }
        }
        if (left >= 0)
        {
            diskRows.Add(row);
            diskColumns.Add(left);
            diskLengths.Add(right - left + 1);
        }
    }
    var diskShape = Operators.ShapeOf(new Region(80, 80, diskRows.ToArray(), diskColumns.ToArray(), diskLengths.ToArray()));
    if (Math.Abs(diskShape.Circularity - 0.9081810736527025) > 1e-4 || Math.Abs(diskShape.Rectangularity - 0.8115961921519181) > 1e-4)
        throw new InvalidOperationException($"圓的圓度、矩形度不對，得到 {diskShape.Circularity} {diskShape.Rectangularity}");

    var barImage = new byte[32 * 64];
    for (int row = 0; row < 32; row++)
        for (int column = 16; column < 48; column++)
            barImage[row * 64 + column] = 255;
    var barWidth = Operators.MeasureWidth(ImageBuffer.Gray(64, 32, barImage), 16);
    if (Math.Abs(barWidth.Left - 15.5) > 1e-6 || Math.Abs(barWidth.Right - 47.5) > 1e-6 || Math.Abs(barWidth.Width - 32) > 1e-6)
        throw new InvalidOperationException($"亮條寬度應是 32，左右在 15.5 和 47.5，得到 {barWidth.Left} {barWidth.Right} {barWidth.Width}");
    var leftEdge = Operators.FitLine(new (double Row, double Col)[] { (0, 15.5), (20, 15.5), (40, 15.5) });
    var rightEdge = Operators.FitLine(new (double Row, double Col)[] { (0, 47.5), (30, 47.5) });
    double edgeGap = Operators.LineDistance(leftEdge, rightEdge);
    if (Math.Abs(edgeGap - 32) > 1e-6)
        throw new InvalidOperationException($"兩條邊緣的距離應是 32，得到 {edgeGap}");
    var speck = new byte[7 * 7];
    for (int row = 1; row <= 3; row++)
        for (int column = 1; column <= 3; column++)
            speck[row * 7 + column] = 180;
    speck[3 * 7 + 3] = 200;
    speck[5 * 7 + 5] = 200;
    var opened = Operators.GrayOpenRectangle(ImageBuffer.Gray(7, 7, speck), 3, 3);
    if (opened.Pixels[3 * 7 + 3] != 180 || opened.Pixels[0] != 0 || opened.Pixels[5 * 7 + 5] != 0)
        throw new InvalidOperationException($"開運算應留下 3x3 的 180，去掉單獨的 200，得到中心 {opened.Pixels[3 * 7 + 3]}、角落 {opened.Pixels[5 * 7 + 5]}");
    var hole = new byte[7 * 7];
    for (int i = 0; i < hole.Length; i++)
        hole[i] = 180;
    hole[3 * 7 + 3] = 0;
    var closed = Operators.GrayCloseRectangle(ImageBuffer.Gray(7, 7, hole), 3, 3);
    if (closed.Pixels[3 * 7 + 3] != 180)
        throw new InvalidOperationException($"閉運算應補上暗點，得到 {closed.Pixels[3 * 7 + 3]}");
    var roundSpecks = new byte[9 * 9];
    for (int row = 2; row <= 4; row++)
        for (int column = 2; column <= 4; column++)
            roundSpecks[row * 9 + column] = 180;
    roundSpecks[4 * 9 + 4] = 200;
    roundSpecks[7 * 9 + 7] = 200;
    var circleOpened = Operators.GrayOpenCircle(ImageBuffer.Gray(9, 9, roundSpecks), 1);
    if (circleOpened.Pixels[3 * 9 + 3] != 180 || circleOpened.Pixels[3 * 9 + 2] != 180 || circleOpened.Pixels[2 * 9 + 3] != 180)
        throw new InvalidOperationException("半徑 1 的圓形開運算應留下十字中心 180");
    if (circleOpened.Pixels[4 * 9 + 4] != 0 || circleOpened.Pixels[7 * 9 + 7] != 0 || circleOpened.Pixels[2 * 9 + 2] != 0)
        throw new InvalidOperationException("半徑 1 應去掉方塊四角和單獨的亮點");

    var distortedDot = new byte[41 * 41];
    distortedDot[20 * 41 + 35] = 255;
    var camera = new Intrinsics(40, 40, 20, 20, -0.5);
    var undistortedImage = Operators.Undistort(ImageBuffer.Gray(41, 41, distortedDot), camera);
    if (undistortedImage.Pixels[20 * 41 + 36] != 184)
        throw new InvalidOperationException($"去畸變後行 36 應是 184，得到 {undistortedImage.Pixels[20 * 41 + 36]}");
    var identityRotation = new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
    var onPlane = Operators.ImageToPlane(20, 30, camera, identityRotation, 0, 0, 200);
    if (Math.Abs(onPlane.X - 51.73040180762383) > 1e-3 || Math.Abs(onPlane.Y) > 1e-6)
        throw new InvalidOperationException($"像素 (20,30) 應落到平面 x 51.730，得到 {onPlane.X:0.000} {onPlane.Y:0.000}");
    var plainCamera = new Intrinsics(40, 40, 20, 20);
    var plainPlane = Operators.ImageToPlane(20, 30, plainCamera, identityRotation, 0, 0, 200);
    if (Math.Abs(plainPlane.X - 50) > 1e-9 || Math.Abs(plainPlane.Y) > 1e-9)
        throw new InvalidOperationException($"沒有畸變時平面 x 應是 50，得到 {plainPlane.X}");

    var turnedBand = new byte[40 * 40];
    for (int row = 16; row <= 23; row++)
        for (int column = 0; column < 40; column++)
            turnedBand[row * 40 + column] = 255;
    var rotatedBand = Operators.RotateNearest(ImageBuffer.Gray(40, 40, turnedBand), 90);
    var turnedWidth = Operators.MeasureWidth(rotatedBand, 20);
    if (Math.Abs(turnedWidth.Width - 8) > 1e-6)
        throw new InvalidOperationException($"轉正後的寬度應是 8，得到 {turnedWidth.Left:0.00} {turnedWidth.Right:0.00} {turnedWidth.Width:0.00}");
    var leftMillimetre = Operators.ImageToPlane(20, turnedWidth.Left, plainCamera, identityRotation, 0, 0, 200, scale: 1);
    var rightMillimetre = Operators.ImageToPlane(20, turnedWidth.Right, plainCamera, identityRotation, 0, 0, 200, scale: 1);
    if (Math.Abs((rightMillimetre.X - leftMillimetre.X) - 40) > 1e-6)
        throw new InvalidOperationException($"8 像素在這個相機下應是 40，得到 {rightMillimetre.X - leftMillimetre.X}");
    var idealBar = new byte[41 * 41];
    for (int row = 0; row < 41; row++)
        for (int column = 16; column <= 23; column++)
            idealBar[row * 41 + column] = 255;
    var warpedBar = new byte[41 * 41];
    for (int row = 0; row < 41; row++)
    {
        for (int column = 0; column < 41; column++)
        {
            double normalizedX = (column - 20.0) / 40.0;
            double normalizedY = (row - 20.0) / 40.0;
            double radius = normalizedX * normalizedX + normalizedY * normalizedY;
            double factor = 1 - 0.5 * radius;
            if (Math.Abs(factor) < 1e-8)
                continue;
            double idealColumn = 20 + 40 * (normalizedX / factor);
            double idealRow = 20 + 40 * (normalizedY / factor);
            int sampleColumn = (int)Math.Round(idealColumn, MidpointRounding.AwayFromZero);
            int sampleRow = (int)Math.Round(idealRow, MidpointRounding.AwayFromZero);
            if ((uint)sampleRow >= 41 || (uint)sampleColumn >= 41)
                continue;
            warpedBar[row * 41 + column] = idealBar[sampleRow * 41 + sampleColumn];
        }
    }
    double planeWidth = Operators.MeasurePlaneWidth(ImageBuffer.Gray(41, 41, warpedBar), 20, camera, identityRotation, 0, 0, 200);
    if (Math.Abs(planeWidth - 40) > 1)
        throw new InvalidOperationException($"去畸變後的寬度應約 40，得到 {planeWidth:0.00}");

    var board = CalibPlate.Parse("5,7,10");
    var objectX = new double[board.CornerCount];
    var objectY = new double[board.CornerCount];
    int markIndex = 0;
    for (int row = 0; row < board.Rows; row++)
    {
        for (int column = 0; column < board.Columns; column++)
        {
            var mark = board.Point(row, column);
            objectX[markIndex] = mark.X;
            objectY[markIndex] = mark.Y;
            markIndex++;
        }
    }
    const double trueFx = 600;
    const double trueFy = 610;
    const double trueCx = 320;
    const double trueCy = 240;
    (double[] U, double[] V) Project(double[,] rotation, double tx, double ty, double tz)
    {
        var columns = new double[objectX.Length];
        var rows = new double[objectX.Length];
        for (int i = 0; i < objectX.Length; i++)
        {
            double x = rotation[0, 0] * objectX[i] + rotation[0, 1] * objectY[i] + tx;
            double y = rotation[1, 0] * objectX[i] + rotation[1, 1] * objectY[i] + ty;
            double z = rotation[2, 0] * objectX[i] + rotation[2, 1] * objectY[i] + tz;
            columns[i] = trueFx * x / z + trueCx;
            rows[i] = trueFy * y / z + trueCy;
        }
        return (columns, rows);
    }
    double yaw = 15 * Math.PI / 180;
    double pitch = 12 * Math.PI / 180;
    double cosYaw = Math.Cos(yaw);
    double sinYaw = Math.Sin(yaw);
    double cosPitch = Math.Cos(pitch);
    double sinPitch = Math.Sin(pitch);
    var identity = new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
    var aroundY = new double[,] { { cosYaw, 0, sinYaw }, { 0, 1, 0 }, { -sinYaw, 0, cosYaw } };
    var aroundX = new double[,] { { 1, 0, 0 }, { 0, cosPitch, -sinPitch }, { 0, sinPitch, cosPitch } };
    var front = Project(identity, -30, -20, 400);
    var turned = Project(aroundY, -30, -20, 420);
    var tilted = Project(aroundX, -30, -20, 430);
    var intrinsics = PlanarCalib.Estimate(objectX, objectY, new[]
    {
        new PlanarCalib.View(front.U, front.V),
        new PlanarCalib.View(turned.U, turned.V),
        new PlanarCalib.View(tilted.U, tilted.V),
    });
    if (Math.Abs(intrinsics.Fx - trueFx) > 1e-3 || Math.Abs(intrinsics.Fy - trueFy) > 1e-3
        || Math.Abs(intrinsics.Cx - trueCx) > 1e-3 || Math.Abs(intrinsics.Cy - trueCy) > 1e-3)
        throw new InvalidOperationException($"內參應接近 600、610、320、240，得到 {intrinsics.Fx:0.00} {intrinsics.Fy:0.00} {intrinsics.Cx:0.00} {intrinsics.Cy:0.00}");
    if (intrinsics.K1 != 0 || intrinsics.K2 != 0 || intrinsics.K3 != 0)
        throw new InvalidOperationException("沒有畸變的點不該估出 k1");
    const double trueK1 = -0.3;
    (double[] U, double[] V) ProjectDistorted(double[,] rotation, double tx, double ty, double tz)
    {
        var columns = new double[objectX.Length];
        var rows = new double[objectX.Length];
        for (int i = 0; i < objectX.Length; i++)
        {
            double x = rotation[0, 0] * objectX[i] + rotation[0, 1] * objectY[i] + tx;
            double y = rotation[1, 0] * objectX[i] + rotation[1, 1] * objectY[i] + ty;
            double z = rotation[2, 0] * objectX[i] + rotation[2, 1] * objectY[i] + tz;
            double idealX = x / z;
            double idealY = y / z;
            double radius = idealX * idealX + idealY * idealY;
            double factor = 1 + trueK1 * radius;
            columns[i] = trueFx * idealX * factor + trueCx;
            rows[i] = trueFy * idealY * factor + trueCy;
        }
        return (columns, rows);
    }
    var distortedFront = ProjectDistorted(identity, -30, -20, 80);
    var distortedTurned = ProjectDistorted(aroundY, -25, -18, 95);
    var distortedTilted = ProjectDistorted(aroundX, -28, -15, 105);
    var distorted = PlanarCalib.EstimateDistorted(objectX, objectY, new[]
    {
        new PlanarCalib.View(distortedFront.U, distortedFront.V),
        new PlanarCalib.View(distortedTurned.U, distortedTurned.V),
        new PlanarCalib.View(distortedTilted.U, distortedTilted.V),
    });
    if (Math.Abs(distorted.K1 - trueK1) > 1e-4
        || Math.Abs(distorted.Fx - trueFx) > 1e-2 || Math.Abs(distorted.Fy - trueFy) > 1e-2
        || Math.Abs(distorted.Cx - trueCx) > 1e-2 || Math.Abs(distorted.Cy - trueCy) > 1e-2)
        throw new InvalidOperationException($"帶畸變的內參應接近 k1 -0.3、焦距 600/610，得到 k1 {distorted.K1:0.000} {distorted.Fx:0.00} {distorted.Fy:0.00} {distorted.Cx:0.00} {distorted.Cy:0.00}");

    ImageBuffer Board(double[,] rotation, double tx, double ty, double tz) =>
        ChessboardPhoto.Render(640, 480, new Intrinsics(trueFx, trueFy, trueCx, trueCy), rotation, tx, ty, tz, 5, 7, 10);
    (double X, double Y) ProjectBoard(double[,] rotation, double tx, double ty, double tz, double x, double y)
    {
        double cameraX = rotation[0, 0] * x + rotation[0, 1] * y + tx;
        double cameraY = rotation[1, 0] * x + rotation[1, 1] * y + ty;
        double cameraZ = rotation[2, 0] * x + rotation[2, 1] * y + tz;
        return (trueFx * cameraX / cameraZ + trueCx, trueFy * cameraY / cameraZ + trueCy);
    }
    PlanarCalib.View MatchBoard(double[,] rotation, double tx, double ty, double tz)
    {
        var found = ChessboardPhoto.FindCorners(Board(rotation, tx, ty, tz), 5, 7);
        var columns = new double[found.Column.Length];
        var rows = new double[found.Row.Length];
        var used = new bool[found.Column.Length];
        double worst = 0;
        string worstText = "";
        for (int rowIndex = 0; rowIndex < 5; rowIndex++)
        {
            for (int columnIndex = 0; columnIndex < 7; columnIndex++)
            {
                var expected = ProjectBoard(rotation, tx, ty, tz, columnIndex * 10, rowIndex * 10);
                int nearest = -1;
                double best = double.PositiveInfinity;
                for (int i = 0; i < found.Column.Length; i++)
                {
                    if (used[i])
                        continue;
                    double dx = found.Column[i] - expected.X;
                    double dy = found.Row[i] - expected.Y;
                    double distance = dx * dx + dy * dy;
                    if (distance < best)
                    {
                        best = distance;
                        nearest = i;
                    }
                }
                if (nearest < 0)
                    throw new InvalidOperationException("棋盤角點沒有對上");
                if (best > worst)
                {
                    worst = best;
                    worstText = $"({rowIndex},{columnIndex}) 差 {Math.Sqrt(best):0.00}，落在 {found.Column[nearest]:0.00},{found.Row[nearest]:0.00}，應在 {expected.X:0.00},{expected.Y:0.00}";
                }
                used[nearest] = true;
                columns[rowIndex * 7 + columnIndex] = found.Column[nearest];
                rows[rowIndex * 7 + columnIndex] = found.Row[nearest];
            }
        }
        if (worst > 0.75 * 0.75)
            throw new InvalidOperationException("棋盤角點偏了：" + worstText);
        return new PlanarCalib.View(columns, rows);
    }
    double photoYaw = 28 * Math.PI / 180;
    double photoPitch = 24 * Math.PI / 180;
    double cosPhotoYaw = Math.Cos(photoYaw);
    double sinPhotoYaw = Math.Sin(photoYaw);
    double cosPhotoPitch = Math.Cos(photoPitch);
    double sinPhotoPitch = Math.Sin(photoPitch);
    var photoAroundY = new double[,] { { cosPhotoYaw, 0, sinPhotoYaw }, { 0, 1, 0 }, { -sinPhotoYaw, 0, cosPhotoYaw } };
    var photoAroundX = new double[,] { { 1, 0, 0 }, { 0, cosPhotoPitch, -sinPhotoPitch }, { 0, sinPhotoPitch, cosPhotoPitch } };
    var photoFront = MatchBoard(identity, -30, -20, 150);
    var photoTurned = MatchBoard(photoAroundY, -30, -20, 170);
    var photoTilted = MatchBoard(photoAroundX, -30, -20, 165);
    var fromPhotos = PlanarCalib.Estimate(objectX, objectY, new[] { photoFront, photoTurned, photoTilted });
    if (Math.Abs(fromPhotos.Fx - trueFx) > 8 || Math.Abs(fromPhotos.Fy - trueFy) > 8
        || Math.Abs(fromPhotos.Cx - trueCx) > 2 || Math.Abs(fromPhotos.Cy - trueCy) > 2)
        throw new InvalidOperationException($"從棋盤圖估出的內參應接近 600、610、320、240，得到 {fromPhotos.Fx:0.00} {fromPhotos.Fy:0.00} {fromPhotos.Cx:0.00} {fromPhotos.Cy:0.00}");

    var block = new byte[64 * 64];
    for (int row = 16; row < 48; row++)
        for (int col = 16; col < 48; col++)
            block[row * 64 + col] = 255;
    var shifted = new byte[96 * 96];
    for (int row = 36; row < 68; row++)
        for (int col = 31; col < 63; col++)
            shifted[row * 96 + col] = 255;
    var squareHits = Operators.FindShape(ImageBuffer.Gray(96, 96, shifted), ImageBuffer.Gray(64, 64, block), 8, 8, 55, 55);
    if (squareHits.Length < 1)
        throw new InvalidOperationException("平移的方塊應找得到");
    var square = squareHits.OrderByDescending(hit => hit.Score).First();
    double squareRow = square.Row - 51.5;
    double squareCol = square.Column - 46.5;
    if (squareRow * squareRow + squareCol * squareCol > 0.45 * 0.45)
        throw new InvalidOperationException($"方塊中心應在列 51.5、行 46.5 的 0.45 像素內，得到 {square.Row:0.000},{square.Column:0.000} 角度 {square.Angle * 180 / Math.PI:0.00} 分 {square.Score:0.00}");
    double squareDegrees = square.Angle * 180 / Math.PI;
    double squareMod = squareDegrees % 90;
    if (squareMod < 0)
        squareMod += 90;
    if (Math.Min(squareMod, 90 - squareMod) > 1)
        throw new InvalidOperationException($"方塊角度應靠近 90 度的倍數，得到 {squareDegrees:0.00}");
    var squareNcc = Operators.FindNcc(ImageBuffer.Gray(96, 96, shifted), ImageBuffer.Gray(64, 64, block), 8, 8, 55, 55);
    if (squareNcc.Length < 1)
        throw new InvalidOperationException("正規相關應找得到平移的方塊");
    var squareCorr = squareNcc.OrderByDescending(hit => hit.Score).First();
    double nccRow = squareCorr.Row - 51.5;
    double nccCol = squareCorr.Column - 46.5;
    if (nccRow * nccRow + nccCol * nccCol > 0.75 * 0.75)
        throw new InvalidOperationException($"正規相關的方塊中心應在列 51.5、行 46.5 的 0.75 像素內，得到 {squareCorr.Row:0.000},{squareCorr.Column:0.000} 角度 {squareCorr.Angle * 180 / Math.PI:0.00} 分 {squareCorr.Score:0.00}");
    double nccDegrees = squareCorr.Angle * 180 / Math.PI;
    double nccMod = nccDegrees % 90;
    if (nccMod < 0)
        nccMod += 90;
    if (Math.Min(nccMod, 90 - nccMod) > 1)
        throw new InvalidOperationException($"正規相關的方塊角度應靠近 90 度的倍數，得到 {nccDegrees:0.00}");
    var inverted = new byte[shifted.Length];
    for (int i = 0; i < shifted.Length; i++)
        inverted[i] = (byte)(255 - shifted[i]);
    var samePolarity = Operators.FindNcc(ImageBuffer.Gray(96, 96, inverted), ImageBuffer.Gray(64, 64, block), 8, 8, 55, 55);
    if (samePolarity.Any(hit =>
        {
            double dRow = hit.Row - 51.5;
            double dCol = hit.Column - 46.5;
            return dRow * dRow + dCol * dCol <= 3 * 3;
        }))
        throw new InvalidOperationException("沒開反相時，黑白對調的方塊不該算對上");
    var flipped = Operators.FindNcc(ImageBuffer.Gray(96, 96, inverted), ImageBuffer.Gray(64, 64, block), 8, 8, 55, 55, ignoreGlobalPolarity: true);
    if (flipped.Length < 1)
        throw new InvalidOperationException("開了反相應找得到黑白對調的方塊");
    var flippedHit = flipped.OrderByDescending(hit => hit.Score).First();
    double flippedRow = flippedHit.Row - 51.5;
    double flippedCol = flippedHit.Column - 46.5;
    if (flippedRow * flippedRow + flippedCol * flippedCol > 0.75 * 0.75)
        throw new InvalidOperationException($"反相後的方塊中心應在 0.75 像素內，得到 {flippedHit.Row:0.000},{flippedHit.Column:0.000} 分 {flippedHit.Score:0.00}");

    var template = Operators.ReadImage(SampleFile("shape_template.png"));
    var scene = Operators.ReadImage(SampleFile("shapes_scene.png"));
    var found = Operators.FindShape(scene, template, 15, 15, 164, 164);
    var expected = new (double Row, double Column, double Angle)[]
    {
        (130.107, 160.439, 0.1545),
        (359.343, 169.994, -0.4662),
        (124.702, 469.511, 0.5583),
        (349.923, 479.0, 1.1242),
    };
    if (found.Length < 4)
        throw new InvalidOperationException("場景裡應找到 4 個，得到 " + found.Length + "：" + string.Join("; ", found.Select(hit => $"{hit.Row:0.00},{hit.Column:0.00},{hit.Angle * 180 / Math.PI:0.00},{hit.Score:0.00}")));
    int fractional = 0;
    foreach (var pose in expected)
    {
        var hit = found.OrderBy(item =>
        {
            double dRow = item.Row - pose.Row;
            double dCol = item.Column - pose.Column;
            return dRow * dRow + dCol * dCol;
        }).First();
        double dRow = hit.Row - pose.Row;
        double dCol = hit.Column - pose.Column;
        double rowPart = Math.Abs(hit.Row - Math.Round(hit.Row));
        double colPart = Math.Abs(hit.Column - Math.Round(hit.Column));
        if (rowPart > 0.02 || colPart > 0.02)
            fractional++;
        double dAngle = hit.Angle - pose.Angle;
        while (dAngle > Math.PI)
            dAngle -= 2 * Math.PI;
        while (dAngle < -Math.PI)
            dAngle += 2 * Math.PI;
        if (dRow * dRow + dCol * dCol > 1.5 * 1.5 || Math.Abs(dAngle) > 1.0 * Math.PI / 180.0)
            throw new InvalidOperationException($"沒有對上 Python 的 ({pose.Row:0.0},{pose.Column:0.0},{pose.Angle * 180 / Math.PI:0.00}°)。C# 得到 {hit.Row:0.00},{hit.Column:0.00},{hit.Angle * 180 / Math.PI:0.00}°");
    }
    if (fractional < 3)
        throw new InvalidOperationException("次像素精修後，四個裡至少三個的列或行要離開整數。得到 " + string.Join("; ", found.Select(item => $"{item.Row:0.00},{item.Column:0.00}")));

    var correlated = Operators.FindNcc(scene, template, 15, 15, 164, 164);
    if (correlated.Length < 4)
        throw new InvalidOperationException("正規相關應找到 4 個，得到 " + correlated.Length + "：" + string.Join("; ", correlated.Select(hit => $"{hit.Row:0.00},{hit.Column:0.00},{hit.Angle * 180 / Math.PI:0.00},{hit.Score:0.00}")));
    foreach (var pose in expected)
    {
        var hit = correlated.OrderBy(item =>
        {
            double dRow = item.Row - pose.Row;
            double dCol = item.Column - pose.Column;
            return dRow * dRow + dCol * dCol;
        }).First();
        double dRow = hit.Row - pose.Row;
        double dCol = hit.Column - pose.Column;
        double dAngle = hit.Angle - pose.Angle;
        while (dAngle > Math.PI)
            dAngle -= 2 * Math.PI;
        while (dAngle < -Math.PI)
            dAngle += 2 * Math.PI;
        if (dRow * dRow + dCol * dCol > 1.0 * 1.0 || Math.Abs(dAngle) > 1.5 * Math.PI / 180.0)
            throw new InvalidOperationException($"正規相關沒有對上 Python 的 ({pose.Row:0.0},{pose.Column:0.0},{pose.Angle * 180 / Math.PI:0.00}°)。C# 得到 {hit.Row:0.00},{hit.Column:0.00},{hit.Angle * 180 / Math.PI:0.00}° 分 {hit.Score:0.00}");
    }

    Expect(GrayConvert.Bgr(10, 20, 30), 22, "灰階 10,20,30");
    Expect(GrayConvert.Bgr(255, 128, 0), 104, "灰階 255,128,0");
    Expect(GrayConvert.Bgr(1, 2, 3), 2, "灰階 1,2,3");
    Expect(GrayConvert.Bgr(0, 0, 255), 76, "灰階 0,0,255");
    Expect(GrayConvert.Bgr(68, 0, 153), 53, "灰階 68,0,153");

    var image = ImageBuffer.Gray(4, 1, new byte[] { 0, 128, 255, 10 });
    var kept = Operators.Threshold(image, 128, 255);
    if (kept.Rows.Length != 1 || kept.Columns[0] != 1 || kept.Lengths[0] != 2)
        throw new InvalidOperationException("相鄰的 128 和 255 應該是同一段行程");
    var empty = Operators.Threshold(image, 5, 4);
    if (empty.Area != 0)
        throw new InvalidOperationException("最小值比最大值大時應該是空區域");
    var above = Operators.Threshold(image, 256, 400);
    if (above.Area != 0)
        throw new InvalidOperationException("門檻高於 255 不該留下像素");

    var diagonal = Operators.Threshold(ImageBuffer.Gray(3, 3, new byte[]
    {
        0, 255, 0,
        255, 0, 0,
        0, 0, 0,
    }), 1, 255);
    var one = Operators.Connection(diagonal);
    if (one.Length != 1 || one[0].Area != 2)
        throw new InvalidOperationException($"斜角應該是 1 個物件，得到 {one.Length}");

    var dots = new byte[7 * 12];
    dots[3 * 12 + 1] = 255;
    dots[3 * 12 + 4] = 255;
    dots[3 * 12 + 10] = 255;
    var parts = Operators.Connection(Operators.Threshold(ImageBuffer.Gray(12, 7, dots), 1, 255));
    if (parts.Length != 3)
        throw new InvalidOperationException($"三個點應該是 3 個物件，得到 {parts.Length}");
    if (parts[0].Columns[0] != 1 || parts[1].Columns[0] != 4 || parts[2].Columns[0] != 10)
        throw new InvalidOperationException("物件沒有依掃描順序排列");
    var dotRegion = Operators.Threshold(ImageBuffer.Gray(12, 7, dots), 1, 255);
    if (Operators.Connection(dotRegion, 3).Length != 2)
        throw new InvalidOperationException("距離 3 應該併成 2 個");
    var merged = Operators.Connection(dotRegion, 6);
    if (merged.Length != 1 || merged[0].Area != 3)
        throw new InvalidOperationException("距離 6 應該併成 1 個");
    if (Operators.Count(parts) != 3)
        throw new InvalidOperationException("三個點的個數不是 3");
    var center = Operators.AreaCenterOf(one)[0];
    if (center.Area != 2 || Math.Abs(center.Row - 0.5) > 1e-9 || Math.Abs(center.Column - 0.5) > 1e-9)
        throw new InvalidOperationException($"斜角重心應是 0.5, 0.5，得到 {center.Row}, {center.Column}");

    var frame = ImageFrame.Gray(3, 3, new byte[9]);
    var dest = new byte[3 * 3 * 4];
    FrameComposer.Render(frame, new ScreenMap(0, 0, 3, 3, 0, 0, 3, 3), Operators.Overlays(one), dest, 3, 3);
    int red = dest[(0 * 3 + 1) * 4 + 2];
    if (red < 200)
        throw new InvalidOperationException($"行程沒有畫上影像，紅色通道是 {red}");
}

static string SampleFile(string name)
{
    var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
    while (dir != null)
    {
        string path = Path.Combine(dir.FullName, "samples", name);
        if (File.Exists(path))
            return path;
        dir = dir.Parent;
    }
    throw new FileNotFoundException($"找不到 samples/{name}。請在存放 samples 的那一層，或它的子目錄裡執行。");
}

static void Compare(string path)
{
    var json = File.ReadAllText(path);
    var file = JsonSerializer.Deserialize<CompareFile>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
        ?? throw new InvalidOperationException("對照檔是空的");
    if (file.Cases == null || file.Cases.Length == 0)
        throw new InvalidOperationException("對照檔沒有案例");
    foreach (var item in file.Cases)
    {
        var image = Operators.ReadImage(item.Path ?? throw new InvalidOperationException("缺少路徑"));
        if (image.Width != item.Width || image.Height != item.Height || image.Channels != item.Channels)
            throw new InvalidOperationException($"{item.Path} 尺寸 {image.Width}x{image.Height}x{image.Channels}，對照是 {item.Width}x{item.Height}x{item.Channels}");
        var expectedPixels = Convert.FromBase64String(item.Pixels ?? "");
        if (expectedPixels.Length != image.Pixels.Length || !expectedPixels.AsSpan().SequenceEqual(image.Pixels))
            throw new InvalidOperationException($"{item.Path} 的像素和 Python 不同");
        var region = Operators.Threshold(image, item.Min, item.Max);
        ExpectSameRuns(region, item.Threshold, item.Path + " 二值化");
        var parts = Operators.Connection(region);
        var expectedParts = item.Components ?? Array.Empty<int[][]>();
        if (parts.Length != expectedParts.Length)
            throw new InvalidOperationException($"{item.Path} 連通 {parts.Length} 個，Python 是 {expectedParts.Length} 個");
        for (int index = 0; index < parts.Length; index++)
            ExpectSameRuns(parts[index], expectedParts[index], item.Path + " 物件 " + index);
        if (Operators.Count(parts) != item.Count)
            throw new InvalidOperationException($"{item.Path} 個數 {Operators.Count(parts)}，Python 是 {item.Count}");
        var centers = Operators.AreaCenterOf(parts);
        var expectedCenters = item.Centers ?? Array.Empty<double[]>();
        if (centers.Length != expectedCenters.Length)
            throw new InvalidOperationException($"{item.Path} 重心 {centers.Length} 個，Python 是 {expectedCenters.Length} 個");
        for (int index = 0; index < centers.Length; index++)
        {
            var expected = expectedCenters[index];
            if (expected.Length != 3 || centers[index].Area != (int)expected[0]
                || Math.Abs(centers[index].Row - expected[1]) > 1e-6
                || Math.Abs(centers[index].Column - expected[2]) > 1e-6)
                throw new InvalidOperationException($"{item.Path} 第 {index} 個重心");
        }
    }
    if (file.Ops != null)
    {
        foreach (var item in file.Ops)
            RunOp(item);
        Console.WriteLine($"對照 {file.Ops.Length} 個區域算子通過");
    }
    Console.WriteLine($"對照 {file.Cases.Length} 張圖通過");
}

static void RunOp(Op item)
{
    string name = item.Name ?? item.Kind ?? "算子";
    switch (item.Kind)
    {
        case "connection":
        {
            var region = RegionFrom(item.Width, item.Height, item.Runs);
            var parts = Operators.Connection(region, item.Distance);
            ExpectComponents(parts, item.Components, name);
            break;
        }
        case "morph":
        {
            var region = RegionFrom(item.Width, item.Height, item.Runs);
            Region result = item.Mode switch
            {
                "dilate" => Operators.DilateRectangle(region, item.KernelWidth, item.KernelHeight),
                "erode" => Operators.ErodeRectangle(region, item.KernelWidth, item.KernelHeight),
                "open" => Operators.OpenRectangle(region, item.KernelWidth, item.KernelHeight),
                "close" => Operators.CloseRectangle(region, item.KernelWidth, item.KernelHeight),
                "fill" => Operators.FillUp(region),
                "complement" => Operators.Complement(region),
                "dilateCircle" => Operators.DilateCircle(region, item.Distance),
                "erodeCircle" => Operators.ErodeCircle(region, item.Distance),
                "openCircle" => Operators.OpenCircle(region, item.Distance),
                "closeCircle" => Operators.CloseCircle(region, item.Distance),
                "inner" => Operators.Boundary(region, "inner"),
                "outer" => Operators.Boundary(region, "outer"),
                _ => throw new InvalidOperationException(name),
            };
            ExpectSameRuns(result, item.Out, name);
            break;
        }
        case "bool":
        {
            var left = RegionFrom(item.Width, item.Height, item.Left);
            var right = RegionFrom(item.Width, item.Height, item.Right);
            Region result = item.Mode switch
            {
                "union" => Operators.Union(left, right),
                "intersection" => Operators.Intersection(left, right),
                "difference" => Operators.Difference(left, right),
                _ => throw new InvalidOperationException(name),
            };
            ExpectSameRuns(result, item.Out, name);
            break;
        }
        case "shape":
        {
            var objects = ObjectsFrom(item.Width, item.Height, item.Objects);
            var kept = Operators.SelectShape(objects, new[] { item.Feature ?? "area" }, "and", new[] { item.Minimum }, new[] { item.Maximum });
            var indexes = item.Kept ?? Array.Empty<int>();
            if (kept.Length != indexes.Length)
                throw new InvalidOperationException($"{name} 留下 {kept.Length} 個，Python 是 {indexes.Length} 個");
            for (int i = 0; i < kept.Length; i++)
                ExpectSameRuns(kept[i], item.Objects![indexes[i]], name + " " + indexes[i]);
            break;
        }
        case "tone":
        {
            var image = ImageBuffer.Gray(item.Width, item.Height, Convert.FromBase64String(item.Pixels ?? ""));
            var objects = ObjectsFrom(item.Width, item.Height, item.Objects);
            var tone = Operators.IntensityOf(objects, image);
            var means = item.Means ?? Array.Empty<double>();
            var deviations = item.Deviations ?? Array.Empty<double>();
            if (tone.Length != means.Length)
                throw new InvalidOperationException($"{name} 亮度個數不同");
            for (int i = 0; i < tone.Length; i++)
            {
                if (Math.Abs(tone[i].Mean - means[i]) > 1e-6 || Math.Abs(tone[i].Deviation - deviations[i]) > 1e-6)
                    throw new InvalidOperationException($"{name} 第 {i} 個亮度 {tone[i].Mean} / {tone[i].Deviation}");
            }
            break;
        }
        case "gray":
        {
            var pixels = Convert.FromBase64String(item.Pixels ?? "");
            var image = item.Channels == 1
                ? ImageBuffer.Gray(item.Width, item.Height, pixels)
                : ImageBuffer.Bgr(item.Width, item.Height, pixels);
            var values = Operators.GrayAt(image, item.Row, item.Column);
            var expected = item.Values ?? Array.Empty<int>();
            if (values.Length != expected.Length)
                throw new InvalidOperationException($"{name} 灰階長度 {values.Length}");
            for (int i = 0; i < values.Length; i++)
            {
                if (values[i] != expected[i])
                    throw new InvalidOperationException($"{name} 灰階 {values[i]}，Python {expected[i]}");
            }
            break;
        }
        case "rect":
        {
            var rect = item.Width == 0
                ? Operators.Rectangle(item.Row1, item.Column1, item.Row2, item.Column2)
                : Operators.Rectangle(ImageBuffer.Gray(item.Width, item.Height, new byte[item.Width * item.Height]), item.Row1, item.Column1, item.Row2, item.Column2);
            ExpectSameRuns(rect, item.Out, name);
            break;
        }
        case "reduce":
        {
            var pixels = Convert.FromBase64String(item.Pixels ?? "");
            var image = item.Channels == 1
                ? ImageBuffer.Gray(item.Width, item.Height, pixels)
                : ImageBuffer.Bgr(item.Width, item.Height, pixels);
            var window = RegionFrom(item.Width, item.Height, item.Runs);
            var reduced = item.Mode == "change" ? Operators.ChangeDomain(image, window) : Operators.ReduceDomain(image, window);
            if (item.Second != null)
                reduced = Operators.ReduceDomain(reduced, RegionFrom(item.Width, item.Height, item.Second));
            if (item.Min != 0 || item.Max != 0)
                ExpectSameRuns(Operators.Threshold(reduced, item.Min, item.Max), item.Out, name);
            if (!string.IsNullOrEmpty(item.Display))
            {
                var shown = reduced.ForDisplay();
                var expected = Convert.FromBase64String(item.Display);
                if (shown.Length != expected.Length || !shown.AsSpan().SequenceEqual(expected))
                    throw new InvalidOperationException($"{name} 的顯示像素和 Python 不同");
            }
            break;
        }
        case "paint":
        {
            var pixels = Convert.FromBase64String(item.Pixels ?? "");
            var image = item.Channels == 1
                ? ImageBuffer.Gray(item.Width, item.Height, pixels)
                : ImageBuffer.Bgr(item.Width, item.Height, pixels);
            if (item.Runs != null)
                image = Operators.ReduceDomain(image, RegionFrom(item.Width, item.Height, item.Runs));
            var area = RegionFrom(item.Width, item.Height, item.Left);
            var painted = item.Mode == "over"
                ? Operators.Overpaint(area, image, item.Gray, item.Feature ?? "fill")
                : Operators.Paint(area, image, item.Gray, item.Feature ?? "fill");
            var expected = Convert.FromBase64String(item.Display ?? "");
            if (!painted.Pixels.AsSpan().SequenceEqual(expected))
                throw new InvalidOperationException($"{name} 塗完的像素和 Python 不同");
            break;
        }
        case "crop":
        {
            var pixels = Convert.FromBase64String(item.Pixels ?? "");
            var image = ImageBuffer.Gray(item.Width, item.Height, pixels);
            if (item.Runs != null)
                image = Operators.ReduceDomain(image, RegionFrom(item.Width, item.Height, item.Runs));
            var cropped = Operators.Crop(image, item.Row1, item.Column1, item.Row2, item.Column2);
            var expected = Convert.FromBase64String(item.Display ?? "");
            if (cropped.Width != item.KernelWidth || cropped.Height != item.KernelHeight || !cropped.Pixels.AsSpan().SequenceEqual(expected))
                throw new InvalidOperationException($"{name} 裁切後的像素和 Python 不同");
            ExpectSameRuns(cropped.Domain ?? Operators.DomainOf(cropped), item.Out, name + " 定義域");
            break;
        }
        case "bounds":
        {
            var objects = ObjectsFrom(item.Width, item.Height, item.Objects);
            var bounds = Operators.Bounds(objects);
            var expected = item.Rects ?? Array.Empty<int[]>();
            if (bounds.Length != expected.Length)
                throw new InvalidOperationException($"{name} 外接矩形個數不同");
            for (int i = 0; i < bounds.Length; i++)
            {
                var box = bounds[i];
                var want = expected[i];
                if (box.Row1 != want[0] || box.Column1 != want[1] || box.Row2 != want[2] || box.Column2 != want[3])
                    throw new InvalidOperationException($"{name} 第 {i} 個外接矩形");
            }
            break;
        }
        default:
            throw new InvalidOperationException("不認識的對照 " + item.Kind);
    }
}

static Region RegionFrom(int width, int height, int[][]? runs)
{
    runs ??= Array.Empty<int[]>();
    var rows = new int[runs.Length];
    var columns = new int[runs.Length];
    var lengths = new int[runs.Length];
    for (int i = 0; i < runs.Length; i++)
    {
        rows[i] = runs[i][0];
        columns[i] = runs[i][1];
        lengths[i] = runs[i][2];
    }
    return new Region(width, height, rows, columns, lengths);
}

static Region[] ObjectsFrom(int width, int height, int[][][]? objects)
{
    objects ??= Array.Empty<int[][]>();
    var regions = new Region[objects.Length];
    for (int i = 0; i < objects.Length; i++)
        regions[i] = RegionFrom(width, height, objects[i]);
    return regions;
}

static void ExpectComponents(Region[] parts, int[][][]? expected, string label)
{
    expected ??= Array.Empty<int[][]>();
    if (parts.Length != expected.Length)
        throw new InvalidOperationException($"{label} 連通 {parts.Length} 個，Python 是 {expected.Length} 個");
    for (int i = 0; i < parts.Length; i++)
        ExpectSameRuns(parts[i], expected[i], label + " 物件 " + i);
}

static void Expect(int actual, int expected, string label)
{
    if (actual != expected)
        throw new InvalidOperationException($"{label} 應是 {expected}，得到 {actual}");
}

static void ExpectSameRuns(Region region, int[][]? expected, string label)
{
    expected ??= Array.Empty<int[]>();
    if (region.Rows.Length != expected.Length)
        throw new InvalidOperationException($"{label} 行程 {region.Rows.Length}，Python {expected.Length}");
    for (int i = 0; i < expected.Length; i++)
    {
        var run = expected[i];
        if (run.Length != 3 || region.Rows[i] != run[0] || region.Columns[i] != run[1] || region.Lengths[i] != run[2])
            throw new InvalidOperationException($"{label} 第 {i} 段");
    }
}

sealed class CompareFile
{
    public Case[]? Cases { get; set; }
    public Op[]? Ops { get; set; }
}

sealed class Op
{
    [JsonPropertyName("op")]
    public string? Kind { get; set; }
    public string? Name { get; set; }
    public string? Mode { get; set; }
    public string? Feature { get; set; }
    public int Width { get; set; }
    public int Height { get; set; }
    public int Channels { get; set; }
    public double Distance { get; set; }
    public int KernelWidth { get; set; }
    public int KernelHeight { get; set; }
    public int Row { get; set; }
    public int Column { get; set; }
    public int Row1 { get; set; }
    public int Column1 { get; set; }
    public int Row2 { get; set; }
    public int Column2 { get; set; }
    public double Minimum { get; set; }
    public double Maximum { get; set; }
    public string? Pixels { get; set; }
    public int[][]? Runs { get; set; }
    public int[][]? Second { get; set; }
    public string? Display { get; set; }
    public int Min { get; set; }
    public int Max { get; set; }
    public int Gray { get; set; }
    public int[][]? Left { get; set; }
    public int[][]? Right { get; set; }
    public int[][]? Out { get; set; }
    public int[][][]? Objects { get; set; }
    public int[][][]? Components { get; set; }
    public int[]? Kept { get; set; }
    public int[]? Values { get; set; }
    public int[][]? Rects { get; set; }
    public double[]? Means { get; set; }
    public double[]? Deviations { get; set; }
}

sealed class Case
{
    public string? Path { get; set; }
    public int Width { get; set; }
    public int Height { get; set; }
    public int Channels { get; set; }
    public string? Pixels { get; set; }
    public int Min { get; set; }
    public int Max { get; set; }
    public int[][]? Threshold { get; set; }
    public int[][][]? Components { get; set; }
    public int Count { get; set; }
    public double[][]? Centers { get; set; }
}
