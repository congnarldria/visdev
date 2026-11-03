namespace Vizion;

/// <summary>
/// 棋盤標定板。'5,7,10' 是內角點 5 列、7 行、方格邊長 10，角點共 35 個。
/// 物點是 X = 行索引 × 邊長、Y = 列索引 × 邊長、Z = 0，和 Python 的 _object_points 相同。
/// OpenCV 的 patternSize 是 (行數, 列數)，也就是 (7, 5)。
/// 這份不估計相機的九個參數，也不讀六角形標定板。
/// </summary>
public readonly record struct Chessboard(int Rows, int Columns, double Square)
{
    public int CornerCount => Rows * Columns;

    public int PatternWidth => Columns;

    public int PatternHeight => Rows;

    public (double X, double Y, double Z) Point(int row, int column)
    {
        if (row < 0 || column < 0 || row >= Rows || column >= Columns)
            throw new ArgumentOutOfRangeException(nameof(row), "角點編號超出這塊棋盤");
        return (column * Square, row * Square, 0);
    }
}

public static class CalibPlate
{
    public static Chessboard Parse(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            throw new ArgumentException("棋盤至少要 2x2 個內角點，邊長要大於 0");
        string raw = text.Trim().Trim('\'', '"');
        string lowered = raw.ToLowerInvariant();
        if (lowered.EndsWith(".cpd") || lowered.EndsWith(".descr") || raw.Contains('/') || raw.Contains('\\'))
            throw new ArgumentException("這份 C# 參考只解釋三個數字的棋盤。描述檔仍由 Python 讀。");
        string numeric = raw.Replace('x', ',').Replace('X', ',').Replace('，', ',');
        var parts = new List<string>();
        foreach (string item in numeric.Split(','))
        {
            string trimmed = item.Trim();
            if (trimmed.Length > 0)
                parts.Add(trimmed);
        }
        if (parts.Count != 3)
            throw new ArgumentException("棋盤要寫成三個數，例如 5,7,10");
        if (!double.TryParse(parts[0], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out double rowValue)
            || !double.TryParse(parts[1], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out double colValue)
            || !double.TryParse(parts[2], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out double square))
            throw new ArgumentException("棋盤要寫成三個數，例如 5,7,10");
        int rows = (int)Math.Round(rowValue, MidpointRounding.ToEven);
        int columns = (int)Math.Round(colValue, MidpointRounding.ToEven);
        if (rows < 2 || columns < 2 || square <= 0)
            throw new ArgumentException("棋盤至少要 2x2 個內角點，邊長要大於 0");
        return new Chessboard(rows, columns, square);
    }
}
