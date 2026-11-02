using System.Text.Json;
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
    Console.WriteLine($"對照 {file.Cases.Length} 張圖通過");
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
