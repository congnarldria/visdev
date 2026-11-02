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
