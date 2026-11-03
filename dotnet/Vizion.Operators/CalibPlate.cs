namespace Vizion;

/// <summary>
/// 棋盤標定板。'5,7,10' 是內角點 5 列、7 行、方格邊長 10，角點共 35 個。
/// 物點是 X = 行索引 × 邊長、Y = 列索引 × 邊長、Z = 0，和 Python 的 _object_points 相同。
/// OpenCV 的 patternSize 是 (行數, 列數)，也就是 (7, 5)。
/// 圓點描述檔只讀標記座標、邊長和極性。尋找標記略過，不做對位。
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

    /// <summary>
    /// 讀矩形圓點板的描述文字，和 Python 的 parse_plate_description 同一套排序。
    /// Y 由大到小、X 由小到大。邊長是第一列相鄰 X 的中位數。不估計相機參數。
    /// </summary>
    public static DotPlate ParseDots(string description)
    {
        if (string.IsNullOrWhiteSpace(description))
            throw new ArgumentException("描述檔缺少列數或行數");
        int? rows = null;
        int? columns = null;
        double zee = 0;
        string polarity = "light";
        int finderCount = 0;
        int findersRead = 0;
        bool readingFinders = false;
        var marks = new List<PlateMark>();
        foreach (string line in description.Replace("\r\n", "\n").Replace('\r', '\n').Split('\n'))
        {
            string raw = line.Trim();
            if (raw.Length == 0)
                continue;
            if (raw.StartsWith("#", StringComparison.Ordinal))
            {
                if (raw.IndexOf("calibration marks", StringComparison.OrdinalIgnoreCase) >= 0)
                    readingFinders = false;
                continue;
            }
            string[] parts = raw.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            string key = parts[0].ToLowerInvariant();
            if (key == "r" && parts.Length == 2)
            {
                rows = (int)double.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture);
                readingFinders = false;
                continue;
            }
            if (key == "c" && parts.Length == 2)
            {
                columns = (int)double.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture);
                readingFinders = false;
                continue;
            }
            if (key == "z" && parts.Length == 2)
            {
                zee = double.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture);
                readingFinders = false;
                continue;
            }
            if (key == "p" && parts.Length >= 2)
            {
                polarity = parts[1].StartsWith("dark", StringComparison.OrdinalIgnoreCase) ? "dark" : "light";
                readingFinders = false;
                continue;
            }
            if (key == "f" && parts.Length == 2)
            {
                finderCount = (int)double.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture);
                readingFinders = finderCount > 0;
                findersRead = 0;
                continue;
            }
            if (key is "o" or "i" or "t")
            {
                readingFinders = false;
                continue;
            }
            if (readingFinders && parts.Length >= 2 && findersRead < finderCount)
            {
                findersRead++;
                if (findersRead >= finderCount)
                    readingFinders = false;
                continue;
            }
            if (parts.Length >= 3)
            {
                marks.Add(new PlateMark(
                    double.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture),
                    double.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture),
                    double.Parse(parts[2], System.Globalization.CultureInfo.InvariantCulture)));
            }
        }
        if (rows is null || columns is null)
            throw new ArgumentException("描述檔缺少列數或行數");
        if (rows < 2 || columns < 2)
            throw new ArgumentException("標定板至少要 2x2 個標記");
        if (marks.Count == 0)
            throw new ArgumentException("描述檔沒有標記座標");
        if (marks.Count != rows * columns)
            throw new ArgumentException($"描述檔有 {marks.Count} 個標記，列數乘行數是 {rows * columns}");
        marks.Sort((left, right) =>
        {
            int byY = right.Y.CompareTo(left.Y);
            return byY != 0 ? byY : left.X.CompareTo(right.X);
        });
        var gaps = new List<double>();
        int cols = columns.Value;
        for (int index = 1; index < cols; index++)
            gaps.Add(Math.Abs(marks[index].X - marks[index - 1].X));
        return new DotPlate(rows.Value, columns.Value, Median(gaps), zee, polarity, marks.ToArray());
    }

    static double Median(List<double> values)
    {
        if (values.Count == 0)
            return 1;
        var copy = values.ToArray();
        Array.Sort(copy);
        int mid = copy.Length / 2;
        if (copy.Length % 2 == 1)
            return copy[mid];
        return 0.5 * (copy[mid - 1] + copy[mid]);
    }
}

/// <summary>描述檔裡的一顆圓點。座標單位沿用檔案，範例是公尺。</summary>
public readonly record struct PlateMark(double X, double Y, double Radius);

/// <summary>矩形圓點板。標記已按 Y 由大到小、X 由小到大排好。不估計相機的九個參數。</summary>
public readonly record struct DotPlate(int Rows, int Columns, double Square, double Z, string Polarity, PlateMark[] Marks)
{
    public int MarkCount => Marks.Length;

    public (double X, double Y, double Z) Point(int index)
    {
        if (index < 0 || index >= Marks.Length)
            throw new ArgumentOutOfRangeException(nameof(index));
        PlateMark mark = Marks[index];
        return (mark.X, mark.Y, Z);
    }
}
