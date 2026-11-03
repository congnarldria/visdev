using System.IO.Compression;
using Vizion;

namespace Vizion.Inspect.Sample;

/// <summary>
/// 一條可在 Visual Studio 直接跑的檢查。讀圖、去掉亮點、用形狀對位、量寬度和孔，再印出合格或不合格。
/// </summary>
static class Program
{
    const int ModelWidth = 100;
    const int ModelHeight = 80;
    const int SceneWidth = 220;
    const int SceneHeight = 180;
    const int SceneRow = 40;
    const int SceneColumn = 55;
    const int BarRow = 20;
    const int BarColumn = 15;
    const int BarHeight = 40;
    const int BarWidth = 70;
    const int HoleRow = 40;
    const int HoleColumn = 32;
    const int HoleRadius = 7;
    const double WidthMin = 69;
    const double WidthMax = 71;

    static int Main()
    {
        string folder = AppContext.BaseDirectory;
        string modelPath = Path.Combine(folder, "model.png");
        string scenePath = Path.Combine(folder, "part.png");
        WriteGrayPng(modelPath, Paint(ModelWidth, ModelHeight, 0, 0, speck: false, quarterTurn: false));
        WriteGrayPng(scenePath, Paint(SceneWidth, SceneHeight, SceneRow, SceneColumn, speck: true, quarterTurn: true));

        var model = Operators.ReadImage(modelPath);
        var scene = Operators.ReadImage(scenePath);
        Console.WriteLine($"讀圖  {Path.GetFileName(scenePath)}  {scene.Width}x{scene.Height}");
        Console.WriteLine($"樣板  {Path.GetFileName(modelPath)}  {model.Width}x{model.Height}");

        var cleaned = Operators.GrayOpenCircle(scene, 1);
        int speck = scene.Pixels[8 * SceneWidth + 8];
        int speckAfter = cleaned.Pixels[8 * SceneWidth + 8];
        Console.WriteLine($"開運算  雜點 {speck} → {speckAfter}");

        var hits = Operators.FindShape(cleaned, model, 1, 1, ModelHeight - 2, ModelWidth - 2);
        if (hits.Length < 1)
            return Fail("場景裡沒有對上樣板");
        var hit = hits[0];
        double degrees = hit.Angle * 180.0 / Math.PI;
        Console.WriteLine($"形狀  列 {hit.Row:0.00}  行 {hit.Column:0.00}  角度 {degrees:0.00} 度  分數 {hit.Score:0.00}");

        double align = -degrees;
        var aligned = Operators.RotateNearest(cleaned, align);
        int localRow = BarRow + 6;
        int localColumn = BarColumn + BarWidth / 2;
        int measureRow = SceneRow + 40 - (localColumn - 50);
        int measureColumn = SceneColumn + 50 + (localRow - 40);
        var turned = Operators.RotatePoint(measureRow, measureColumn, align, (SceneHeight - 1) / 2.0, (SceneWidth - 1) / 2.0);
        int row = (int)Math.Round(turned.Row);
        if (row < 0 || row >= aligned.Height)
            return Fail($"對位後的量測列 {row} 跑出圖外");
        var width = Operators.MeasureWidth(aligned, row);
        var planeCamera = new Intrinsics(100, 100, (aligned.Width - 1) / 2.0, (aligned.Height - 1) / 2.0);
        var planeRotation = new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        var leftPlane = Operators.ImageToPlane(row, width.Left, planeCamera, planeRotation, 0, 0, 200);
        var rightPlane = Operators.ImageToPlane(row, width.Right, planeCamera, planeRotation, 0, 0, 200);
        double millimetres = Math.Abs(rightPlane.X - leftPlane.X);
        Console.WriteLine($"對位  轉 {align:0.00} 度後，列 {row} 的寬度 {width.Width:0.00} 像素  左 {width.Left:0.00}  右 {width.Right:0.00}");
        Console.WriteLine($"平面  焦距 100、距離 200，寬度 {millimetres:0.00}");

        var dark = Operators.Threshold(cleaned, 0, 20);
        var holes = Operators.Connection(dark);
        Region? hole = null;
        foreach (var part in holes)
        {
            if (part.Area < 20)
                continue;
            hole = part;
            break;
        }
        if (hole == null)
            return Fail("沒有找到孔");
        var shape = Operators.ShapeOf(hole);
        var center = Operators.AreaCenterOf(hole)[0];
        var centerOnImage = Operators.RotatePoint(center.Row, center.Column, align, (SceneHeight - 1) / 2.0, (SceneWidth - 1) / 2.0);
        var centerOnPlane = Operators.ImageToPlane(centerOnImage.Row, centerOnImage.Column, planeCamera, planeRotation, 0, 0, 200);
        Console.WriteLine($"孔    面積 {hole.Area}  圓度 {shape.Circularity:0.00}  平面 ({centerOnPlane.X:0.00}, {centerOnPlane.Y:0.00})");

        bool widthOk = width.Width >= WidthMin && width.Width <= WidthMax && Math.Abs(millimetres - 140) <= 1;
        bool holeOk = hole.Area >= 120 && hole.Area <= 180 && shape.Circularity >= 0.7;
        double angleError = Math.Abs(Math.Abs(degrees) - 90);
        bool placeOk = angleError <= 5 && Math.Abs(hit.Row - (40.5 + SceneRow)) <= 6 && Math.Abs(hit.Column - (49.5 + SceneColumn)) <= 6;
        bool scoreOk = hit.Score >= 0.5;
        bool speckOk = speckAfter == 40;
        string judge = widthOk && holeOk && placeOk && scoreOk && speckOk ? "合格" : "不合格";
        Console.WriteLine($"判定  {judge}");
        if (judge != "合格")
            return Fail($"寬度 {widthOk}  孔 {holeOk}  位置 {placeOk}  分數 {scoreOk}  雜點 {speckOk}");
        return 0;
    }

    static int Fail(string message)
    {
        Console.WriteLine("範例沒有得到合格：" + message);
        return 1;
    }

    readonly record struct Plate(int Width, int Height, byte[] Pixels);

    static Plate Paint(int width, int height, int originRow, int originColumn, bool speck, bool quarterTurn)
    {
        var pixels = new byte[width * height];
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = 40;
        void Put(int localRow, int localColumn, byte value)
        {
            int row = originRow + localRow;
            int column = originColumn + localColumn;
            if (quarterTurn)
            {
                int fromCenterRow = localRow - 40;
                int fromCenterColumn = localColumn - 50;
                row = originRow + 40 - fromCenterColumn;
                column = originColumn + 50 + fromCenterRow;
            }
            if ((uint)row >= (uint)height || (uint)column >= (uint)width)
                return;
            pixels[row * width + column] = value;
        }
        for (int row = 0; row < BarHeight; row++)
        {
            for (int column = 0; column < BarWidth; column++)
                Put(BarRow + row, BarColumn + column, 220);
        }
        for (int row = -HoleRadius; row <= HoleRadius; row++)
        {
            for (int column = -HoleRadius; column <= HoleRadius; column++)
            {
                if (row * row + column * column > HoleRadius * HoleRadius)
                    continue;
                Put(HoleRow + row, HoleColumn + column, 0);
            }
        }
        if (speck)
            pixels[8 * width + 8] = 255;
        return new Plate(width, height, pixels);
    }

    static void WriteGrayPng(string path, Plate plate)
    {
        using var raw = new MemoryStream();
        for (int row = 0; row < plate.Height; row++)
        {
            raw.WriteByte(0);
            raw.Write(plate.Pixels, row * plate.Width, plate.Width);
        }
        raw.Position = 0;
        using var compressed = new MemoryStream();
        using (var zlib = new ZLibStream(compressed, CompressionLevel.SmallestSize, leaveOpen: true))
            raw.CopyTo(zlib);
        byte[] idat = compressed.ToArray();
        using var file = new MemoryStream();
        file.Write(new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 });
        WriteChunk(file, "IHDR", Header(plate.Width, plate.Height));
        WriteChunk(file, "IDAT", idat);
        WriteChunk(file, "IEND", Array.Empty<byte>());
        File.WriteAllBytes(path, file.ToArray());
    }

    static byte[] Header(int width, int height)
    {
        var data = new byte[13];
        WriteInt(data, 0, width);
        WriteInt(data, 4, height);
        data[8] = 8;
        return data;
    }

    static void WriteChunk(Stream file, string type, byte[] data)
    {
        var head = new byte[4 + data.Length];
        for (int i = 0; i < 4; i++)
            head[i] = (byte)type[i];
        Buffer.BlockCopy(data, 0, head, 4, data.Length);
        WriteInt(file, data.Length);
        file.Write(head);
        WriteInt(file, (int)Crc(head));
    }

    static void WriteInt(Stream stream, int value)
    {
        stream.WriteByte((byte)(value >> 24));
        stream.WriteByte((byte)(value >> 16));
        stream.WriteByte((byte)(value >> 8));
        stream.WriteByte((byte)value);
    }

    static void WriteInt(byte[] data, int offset, int value)
    {
        data[offset] = (byte)(value >> 24);
        data[offset + 1] = (byte)(value >> 16);
        data[offset + 2] = (byte)(value >> 8);
        data[offset + 3] = (byte)value;
    }

    static uint Crc(byte[] data)
    {
        uint crc = 0xFFFFFFFF;
        foreach (byte item in data)
        {
            crc ^= item;
            for (int bit = 0; bit < 8; bit++)
                crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xEDB88320 : crc >> 1;
        }
        return ~crc;
    }
}
