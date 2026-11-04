using System.Globalization;

namespace Photonix;

/// <summary>和網頁版 geom.js 的繪形同一套。線保留起點到終點，不改成外接框。</summary>
static class ShapeGeometry
{
    public static readonly string[] Names =
    [
        "Row", "Column", "Row1", "Column1", "Row2", "Column2",
        "Radius", "Radius1", "Radius2", "Phi", "Length1", "Length2",
    ];

    public static List<string> Tools(IEnumerable<string> names)
    {
        var has = new HashSet<string>(names, StringComparer.Ordinal);
        var tools = new List<string>();
        if (has.Contains("Row1") && has.Contains("Column1") && has.Contains("Row2") && has.Contains("Column2"))
        {
            tools.Add("line");
            tools.Add("rect");
        }
        if (has.Contains("Row") && has.Contains("Column") && has.Contains("Length1") && has.Contains("Length2") && !tools.Contains("rect"))
            tools.Add("rect");
        if (has.Contains("Row") && has.Contains("Column") && has.Contains("Radius"))
            tools.Add("circle");
        if (has.Contains("Row") && has.Contains("Column") && has.Contains("Radius1") && has.Contains("Radius2"))
            tools.Add("ellipse");
        return tools;
    }

    public static Dictionary<string, double> Numbers(string tool, double aRow, double aCol, double bRow, double bCol, string mode)
    {
        (double col1, double col2, double row1, double row2, double width, double height) = Box(aCol, aRow, bCol, bRow);
        if (tool == "line")
            return new Dictionary<string, double> { ["Row1"] = aRow, ["Column1"] = aCol, ["Row2"] = bRow, ["Column2"] = bCol };
        if (tool == "rect" && mode == "corners")
            return new Dictionary<string, double> { ["Row1"] = row1, ["Column1"] = col1, ["Row2"] = row2, ["Column2"] = col2 };
        if (tool == "rect")
        {
            return new Dictionary<string, double>
            {
                ["Row"] = (row1 + row2) / 2,
                ["Column"] = (col1 + col2) / 2,
                ["Phi"] = 0,
                ["Length1"] = width / 2,
                ["Length2"] = height / 2,
            };
        }
        if (tool == "circle")
        {
            double dc = bCol - aCol;
            double dr = bRow - aRow;
            return new Dictionary<string, double>
            {
                ["Row"] = aRow,
                ["Column"] = aCol,
                ["Radius"] = Math.Sqrt(dc * dc + dr * dr),
            };
        }
        double along = width / 2;
        double across = height / 2;
        bool inner = mode == "inner";
        return new Dictionary<string, double>
        {
            ["Row"] = (row1 + row2) / 2,
            ["Column"] = (col1 + col2) / 2,
            ["Phi"] = 0,
            ["Radius1"] = inner ? Math.Min(along, across) : along,
            ["Radius2"] = inner ? Math.Max(along, across) : across,
        };
    }

    public static string Describe(string tool, double aRow, double aCol, double bRow, double bCol, bool corners)
    {
        (double col1, double col2, double row1, double row2, double width, double height) = Box(aCol, aRow, bCol, bRow);
        if (tool == "line")
        {
            double dc = bCol - aCol;
            double dr = bRow - aRow;
            double length = Math.Sqrt(dc * dc + dr * dr);
            return "卡尺  列 " + Format(aRow) + " 行 " + Format(aCol) + " → 列 " + Format(bRow) + " 行 " + Format(bCol) + "  長度 " + Format(length);
        }
        if (tool == "circle")
        {
            double dc = bCol - aCol;
            double dr = bRow - aRow;
            return "圓  列 " + Format(aRow) + "  行 " + Format(aCol) + "  半徑 " + Format(Math.Sqrt(dc * dc + dr * dr));
        }
        if (tool == "ellipse")
            return "橢圓  列 " + Format((row1 + row2) / 2) + "  行 " + Format((col1 + col2) / 2) + "  半徑 " + Format(width / 2) + " × " + Format(height / 2);
        if (corners)
            return "矩形  列 " + Format(row1) + "–" + Format(row2) + "  行 " + Format(col1) + "–" + Format(col2);
        return "矩形  列 " + Format((row1 + row2) / 2) + "  行 " + Format((col1 + col2) / 2) + "  半長 " + Format(width / 2) + " × " + Format(height / 2);
    }

    public static string Format(double value)
    {
        double rounded = Math.Round(value * 100, MidpointRounding.ToPositiveInfinity) / 100;
        return rounded.ToString("0.##", CultureInfo.InvariantCulture);
    }

    public static string Title(string tool) => tool switch
    {
        "line" => "線",
        "rect" => "框選",
        "circle" => "圓",
        _ => "橢圓",
    };

    static (double Col1, double Col2, double Row1, double Row2, double Width, double Height) Box(double aCol, double aRow, double bCol, double bRow)
    {
        double col1 = Math.Min(aCol, bCol);
        double col2 = Math.Max(aCol, bCol);
        double row1 = Math.Min(aRow, bRow);
        double row2 = Math.Max(aRow, bRow);
        return (col1, col2, row1, row2, col2 - col1, row2 - row1);
    }
}
