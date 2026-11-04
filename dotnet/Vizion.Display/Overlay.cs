namespace Vizion;

public enum OverlayKind
{
    Rectangle,
    Polyline,
    Runs,
    Cross,
}

/// <summary>疊在影像上的圖形。座標是影像的列、行，不是控制項像素。</summary>
public sealed class Overlay
{
    public OverlayKind Kind { get; private init; }
    public uint Color { get; private init; }
    public bool Filled { get; private init; }
    public double Row1 { get; private init; }
    public double Column1 { get; private init; }
    public double Row2 { get; private init; }
    public double Column2 { get; private init; }
    public double Size { get; private init; }
    public double[] Rows { get; private init; } = [];
    public double[] Columns { get; private init; } = [];
    public int[] RunRows { get; private init; } = [];
    public int[] RunColumns { get; private init; } = [];
    public int[] RunLengths { get; private init; } = [];

    /// <summary>輪廓和十字的線寬，單位是畫面像素。1 到 10。區域填色不用這個值。</summary>
    public int Stroke { get; private init; } = 1;

    public Overlay WithStroke(int stroke) => new()
    {
        Kind = Kind,
        Color = Color,
        Filled = Filled,
        Row1 = Row1,
        Column1 = Column1,
        Row2 = Row2,
        Column2 = Column2,
        Size = Size,
        Rows = Rows,
        Columns = Columns,
        RunRows = RunRows,
        RunColumns = RunColumns,
        RunLengths = RunLengths,
        Stroke = stroke < 1 ? 1 : stroke,
    };

    public static Overlay Rectangle(double row1, double column1, double row2, double column2, uint color, bool filled = false) =>
        new()
        {
            Kind = OverlayKind.Rectangle,
            Row1 = row1,
            Column1 = column1,
            Row2 = row2,
            Column2 = column2,
            Color = color,
            Filled = filled,
        };

    public static Overlay Polyline(double[] rows, double[] columns, uint color)
    {
        if (rows.Length != columns.Length)
            throw new ArgumentException("列和行的點數要相同");
        return new Overlay { Kind = OverlayKind.Polyline, Rows = rows, Columns = columns, Color = color };
    }

    /// <summary>區域的行程編碼。每一段是同一列上從 column 開始、長度 length 的像素。</summary>
    public static Overlay Runs(int[] rows, int[] columns, int[] lengths, uint color)
    {
        if (rows.Length != columns.Length || rows.Length != lengths.Length)
            throw new ArgumentException("行程的列、行、長度要一樣長");
        return new Overlay
        {
            Kind = OverlayKind.Runs,
            RunRows = rows,
            RunColumns = columns,
            RunLengths = lengths,
            Color = color,
        };
    }

    public static Overlay Cross(double row, double column, double size, uint color) =>
        new() { Kind = OverlayKind.Cross, Row1 = row, Column1 = column, Size = size, Color = color };
}
