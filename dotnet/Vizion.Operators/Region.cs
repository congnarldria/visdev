namespace Vizion;

/// <summary>一個區域，用水平行程表示。每一段是同一列上從欄開始的連續前景。</summary>
public sealed class Region
{
    public int Width { get; }
    public int Height { get; }
    public int[] Rows { get; }
    public int[] Columns { get; }
    public int[] Lengths { get; }

    public int Area
    {
        get
        {
            int area = 0;
            for (int i = 0; i < Lengths.Length; i++)
                area += Lengths[i];
            return area;
        }
    }

    public Region(int width, int height, int[] rows, int[] columns, int[] lengths)
    {
        if (width <= 0 || height <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), "寬高至少要 1");
        if (rows == null || columns == null || lengths == null)
            throw new ArgumentNullException(nameof(rows));
        if (rows.Length != columns.Length || rows.Length != lengths.Length)
            throw new ArgumentException("行程的列、行、長度要一樣長");
        Width = width;
        Height = height;
        Rows = rows;
        Columns = columns;
        Lengths = lengths;
    }

    public Overlay ToOverlay(uint color) => Overlay.Runs(Rows, Columns, Lengths, color);
}
