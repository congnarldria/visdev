using System.Globalization;
using System.IO;
using System.Text.Json;
using Vizion;

namespace Photonix;

/// <summary>點到的一個區域或輪廓。索引從 0 起，畫面顯示時加 1。</summary>
readonly record struct FeatureHit(string Name, int Index, string Kind);

/// <summary>和網頁版特徵視窗同一套目錄。數值只算點到的那一個。</summary>
static class FeaturePick
{
    public static readonly string[] RegionNames =
    {
        "area", "row", "column", "width", "height", "ratio",
        "circularity", "compactness", "contlength", "rectangularity", "convexity",
        "ra", "rb", "phi", "anisometry", "bulkiness", "struct_factor",
        "outer_radius", "max_diameter", "orientation",
        "rect2_phi", "rect2_len1", "rect2_len2",
        "holes_num", "area_holes", "roundness", "num_sides", "euler_number",
        "row1", "column1", "row2", "column2", "connect_num",
    };

    public static readonly string[] GrayNames =
    {
        "gray_min", "gray_max", "gray_mean", "gray_deviation", "gray_median",
    };

    public static readonly string[] XldNames =
    {
        "area", "row", "column", "width", "height", "ratio",
        "circularity", "compactness", "contlength", "rectangularity", "convexity",
        "ra", "rb", "phi", "anisometry", "bulkiness", "struct_factor",
        "outer_radius", "max_diameter", "orientation",
        "rect2_phi", "rect2_len1", "rect2_len2",
    };

    public static readonly Dictionary<string, string> GrayLabels = new(StringComparer.Ordinal)
    {
        ["gray_min"] = "最小輝度",
        ["gray_max"] = "最大輝度",
        ["gray_mean"] = "平均輝度",
        ["gray_deviation"] = "輝度標準差",
        ["gray_median"] = "輝度中位數",
    };

    public static readonly string[] Defaults =
    {
        "region:area", "region:row", "region:column", "region:circularity", "region:contlength",
        "gray:gray_mean", "gray:gray_deviation",
        "xld:contlength", "subpixel",
    };

    public static FeatureHit? Find(IReadOnlyDictionary<string, object> vars, double column, double row, double scale)
    {
        FeatureHit? regionHit = null;
        FeatureHit? xldHit = null;
        double xldDistance = 12 / Math.Max(scale, 0.05);
        int pixelRow = (int)Math.Floor(row);
        int pixelColumn = (int)Math.Floor(column);
        foreach (string name in vars.Keys.Reverse())
        {
            switch (vars[name])
            {
                case Region region:
                    ConsiderRegion(name, 0, region, pixelRow, pixelColumn, ref regionHit);
                    break;
                case Region[] many:
                    for (int i = many.Length - 1; i >= 0; i--)
                        ConsiderRegion(name, i, many[i], pixelRow, pixelColumn, ref regionHit);
                    break;
                case SubpixFigure figure:
                    for (int i = 0; i < figure.Contours.Length; i++)
                    {
                        double distance = LineDistance(column, row, figure.Contours[i]);
                        if (distance <= xldDistance)
                        {
                            xldDistance = distance;
                            xldHit = new FeatureHit(name, figure.Groups[i], "xld");
                        }
                    }
                    for (int i = figure.Objects.Length - 1; i >= 0; i--)
                        ConsiderRegion(name, i, figure.Objects[i], pixelRow, pixelColumn, ref regionHit);
                    break;
                case EdgePoint[][] edges:
                    for (int i = 0; i < edges.Length; i++)
                    {
                        double distance = LineDistance(column, row, edges[i]);
                        if (distance <= xldDistance)
                        {
                            xldDistance = distance;
                            xldHit = new FeatureHit(name, i, "xld");
                        }
                    }
                    break;
            }
        }
        return xldHit ?? regionHit;
    }

    public static Dictionary<string, double> Measure(IReadOnlyDictionary<string, object> vars, FeatureHit hit, ImageBuffer? image)
    {
        var values = new Dictionary<string, double>(StringComparer.Ordinal);
        if (hit.Kind == "xld")
        {
            EdgePoint[]? chain = Chain(vars, hit);
            if (chain != null)
                AddXld(values, chain);
            return values;
        }
        Region? region = RegionAt(vars, hit);
        if (region == null)
            return values;
        AddRegion(values, region, image);
        EdgePoint[] outline = Operators.OuterPoints(region);
        if (outline.Length >= 2)
            AddXld(values, outline);
        return values;
    }

    public static EdgePoint[]? Outline(IReadOnlyDictionary<string, object> vars, FeatureHit hit)
    {
        if (hit.Kind == "xld")
            return Chain(vars, hit);
        Region? region = RegionAt(vars, hit);
        return region == null ? null : Operators.OuterPoints(region);
    }

    public static HashSet<string> LoadKeys()
    {
        try
        {
            if (File.Exists(Path))
            {
                string[]? saved = JsonSerializer.Deserialize<string[]>(File.ReadAllText(Path));
                if (saved != null)
                    return new HashSet<string>(saved, StringComparer.Ordinal);
            }
        }
        catch (IOException)
        {
        }
        catch (JsonException)
        {
        }
        return new HashSet<string>(Defaults, StringComparer.Ordinal);
    }

    public static void SaveKeys(IEnumerable<string> keys)
    {
        Directory.CreateDirectory(System.IO.Path.GetDirectoryName(Path)!);
        File.WriteAllText(Path, JsonSerializer.Serialize(keys.ToArray()));
    }

    public static Overlay[] Dashed(EdgePoint[] chain, double zoom)
    {
        if (chain.Length < 2)
            return [];
        double scale = Math.Max(zoom, 0.05);
        var overlays = new List<Overlay>();
        AddDashed(overlays, chain, 7 / scale, 5 / scale, 0xFF000000, 4);
        AddDashed(overlays, chain, 7 / scale, 5 / scale, 0xFFFFFFFF, 2);
        return overlays.ToArray();
    }

    static void AddDashed(List<Overlay> overlays, EdgePoint[] chain, double dash, double gap, uint color, int stroke)
    {
        bool on = true;
        double drawn = 0;
        var rows = new List<double> { chain[0].Row };
        var columns = new List<double> { chain[0].Column };
        void Flush()
        {
            if (rows.Count >= 2)
                overlays.Add(Overlay.Polyline(rows.ToArray(), columns.ToArray(), color).WithStroke(stroke));
            rows.Clear();
            columns.Clear();
        }
        for (int i = 1; i < chain.Length; i++)
        {
            double row0 = chain[i - 1].Row;
            double column0 = chain[i - 1].Column;
            double row1 = chain[i].Row;
            double column1 = chain[i].Column;
            double dr = row1 - row0;
            double dc = column1 - column0;
            double length = Math.Sqrt(dr * dr + dc * dc);
            if (length < 1e-9)
                continue;
            double walked = 0;
            while (walked < length - 1e-9)
            {
                double room = (on ? dash : gap) - drawn;
                double step = Math.Min(room, length - walked);
                walked += step;
                drawn += step;
                double t = walked / length;
                double row = row0 + dr * t;
                double column = column0 + dc * t;
                if (on)
                {
                    rows.Add(row);
                    columns.Add(column);
                }
                if (drawn >= (on ? dash : gap) - 1e-9)
                {
                    if (on)
                        Flush();
                    on = !on;
                    drawn = 0;
                    if (on)
                    {
                        rows.Add(row);
                        columns.Add(column);
                    }
                }
            }
        }
        if (on)
            Flush();
    }

    public static string Format(double value)
    {
        double rounded = Math.Round(value, 4, MidpointRounding.ToEven);
        if (Math.Abs(rounded - Math.Round(rounded)) < 1e-9)
            return Math.Round(rounded).ToString(CultureInfo.InvariantCulture);
        return rounded.ToString("0.####", CultureInfo.InvariantCulture);
    }

    static void AddRegion(Dictionary<string, double> values, Region region, ImageBuffer? image)
    {
        Operators.AreaCenter center = Operators.AreaCenterOf(region)[0];
        values["region:area"] = center.Area;
        values["region:row"] = center.Row;
        values["region:column"] = center.Column;
        AxisRectangle[] boxes = Operators.Bounds(region);
        if (boxes.Length > 0)
        {
            AxisRectangle box = boxes[0];
            values["region:row1"] = box.Row1;
            values["region:column1"] = box.Column1;
            values["region:row2"] = box.Row2;
            values["region:column2"] = box.Column2;
            values["region:width"] = box.Width;
            values["region:height"] = box.Height;
            values["region:ratio"] = box.Height == 0 ? 0 : (double)box.Width / box.Height;
        }
        RegionShape shape = Operators.ShapeOf(region);
        values["region:circularity"] = shape.Circularity;
        values["region:rectangularity"] = shape.Rectangularity;
        values["region:ra"] = shape.Ra;
        values["region:rb"] = shape.Rb;
        values["region:phi"] = shape.Orientation;
        values["region:orientation"] = shape.Orientation;
        values["region:anisometry"] = shape.Rb > 1e-6 ? shape.Ra / shape.Rb : 0;
        values["region:bulkiness"] = center.Area > 0 ? Math.PI * shape.Ra * shape.Rb / center.Area : 0;
        values["region:struct_factor"] = values["region:anisometry"] * values["region:bulkiness"] - 1;
        double length = Operators.BoundaryLength(region);
        values["region:contlength"] = length;
        values["region:compactness"] = center.Area <= 0 ? 0 : length * length / (4 * Math.PI * center.Area);
        Region filled = Operators.FillUp(region);
        Region holes = Operators.Difference(filled, region);
        int holeCount = holes.Area == 0 ? 0 : Operators.Connection(holes).Length;
        values["region:holes_num"] = holeCount;
        values["region:area_holes"] = holes.Area;
        values["region:euler_number"] = 1 - holeCount;
        values["region:connect_num"] = center.Area == 0 ? 0 : Operators.Connection(region).Length;
        if (image == null || image.Width != region.Width || image.Height != region.Height)
            return;
        AddGray(values, region, image);
    }

    static void AddGray(Dictionary<string, double> values, Region region, ImageBuffer image)
    {
        byte[] gray = image.GrayBytes();
        int count = 0;
        int min = 255;
        int max = 0;
        var samples = new List<byte>();
        for (int i = 0; i < region.Rows.Length; i++)
        {
            int offset = region.Rows[i] * image.Width + region.Columns[i];
            int length = region.Lengths[i];
            for (int k = 0; k < length; k++)
            {
                byte value = gray[offset + k];
                samples.Add(value);
                if (value < min)
                    min = value;
                if (value > max)
                    max = value;
                count++;
            }
        }
        if (count == 0)
            return;
        Intensity stats = Operators.IntensityOf(region, image)[0];
        samples.Sort();
        double median = count % 2 == 1
            ? samples[count / 2]
            : (samples[count / 2 - 1] + samples[count / 2]) / 2.0;
        values["gray:gray_min"] = min;
        values["gray:gray_max"] = max;
        values["gray:gray_mean"] = stats.Mean;
        values["gray:gray_deviation"] = stats.Deviation;
        values["gray:gray_median"] = median;
    }

    static void AddXld(Dictionary<string, double> values, EdgePoint[] chain)
    {
        if (chain.Length < 2)
            return;
        double length = 0;
        double area2 = 0;
        double minRow = chain[0].Row;
        double maxRow = chain[0].Row;
        double minColumn = chain[0].Column;
        double maxColumn = chain[0].Column;
        for (int i = 1; i < chain.Length; i++)
        {
            double dr = chain[i].Row - chain[i - 1].Row;
            double dc = chain[i].Column - chain[i - 1].Column;
            length += Math.Sqrt(dr * dr + dc * dc);
            area2 += chain[i - 1].Column * chain[i].Row - chain[i].Column * chain[i - 1].Row;
            if (chain[i].Row < minRow)
                minRow = chain[i].Row;
            if (chain[i].Row > maxRow)
                maxRow = chain[i].Row;
            if (chain[i].Column < minColumn)
                minColumn = chain[i].Column;
            if (chain[i].Column > maxColumn)
                maxColumn = chain[i].Column;
        }
        double area = Math.Abs(area2) / 2;
        values["xld:contlength"] = length;
        values["xld:area"] = area;
        values["xld:width"] = maxColumn - minColumn;
        values["xld:height"] = maxRow - minRow;
        values["xld:ratio"] = values["xld:height"] == 0 ? 0 : values["xld:width"] / values["xld:height"];
        values["xld:circularity"] = length <= 0 || area <= 0 ? 0 : Math.Min(1, 4 * Math.PI * area / (length * length));
        values["xld:compactness"] = area <= 0 ? 0 : length * length / (4 * Math.PI * area);
    }

    static void ConsiderRegion(string name, int index, Region region, int row, int column, ref FeatureHit? hit)
    {
        if (hit != null || Operators.Contains(region, row, column) != 1)
            return;
        hit = new FeatureHit(name, index, "region");
    }

    static Region? RegionAt(IReadOnlyDictionary<string, object> vars, FeatureHit hit)
    {
        if (!vars.TryGetValue(hit.Name, out object? value))
            return null;
        return value switch
        {
            Region region when hit.Index == 0 => region,
            Region[] many when (uint)hit.Index < (uint)many.Length => many[hit.Index],
            SubpixFigure figure when (uint)hit.Index < (uint)figure.Objects.Length => figure.Objects[hit.Index],
            _ => null,
        };
    }

    static EdgePoint[]? Chain(IReadOnlyDictionary<string, object> vars, FeatureHit hit)
    {
        if (!vars.TryGetValue(hit.Name, out object? value))
            return null;
        if (value is EdgePoint[][] edges && (uint)hit.Index < (uint)edges.Length)
            return edges[hit.Index];
        if (value is not SubpixFigure figure)
            return null;
        EdgePoint[]? longest = null;
        for (int i = 0; i < figure.Contours.Length; i++)
        {
            if (figure.Groups[i] != hit.Index)
                continue;
            if (longest == null || figure.Contours[i].Length > longest.Length)
                longest = figure.Contours[i];
        }
        return longest;
    }

    static double LineDistance(double column, double row, EdgePoint[] line)
    {
        double best = double.PositiveInfinity;
        for (int i = 1; i < line.Length; i++)
        {
            double ax = line[i - 1].Column;
            double ay = line[i - 1].Row;
            double bx = line[i].Column;
            double by = line[i].Row;
            double dx = bx - ax;
            double dy = by - ay;
            double len2 = dx * dx + dy * dy;
            if (len2 < 1e-12)
                len2 = 1;
            double t = ((column - ax) * dx + (row - ay) * dy) / len2;
            if (t < 0)
                t = 0;
            else if (t > 1)
                t = 1;
            double px = ax + t * dx;
            double py = ay + t * dy;
            double dist = (column - px) * (column - px) + (row - py) * (row - py);
            if (dist < best)
                best = dist;
        }
        return Math.Sqrt(best);
    }

    static string Path => System.IO.Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Photonix",
        "features.json");
}
