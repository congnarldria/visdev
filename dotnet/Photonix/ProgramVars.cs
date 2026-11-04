using System.Globalization;
using Vizion;

namespace Photonix;

sealed class VarText
{
    public string Name { get; init; } = "";
    public bool Iconic { get; init; }
    public string Kind { get; init; } = "";
    public string Info { get; init; } = "";
    public string Short { get; init; } = "";
    public string Full { get; init; } = "";
}

static class ProgramVars
{
    public static List<VarText> Describe(IReadOnlyDictionary<string, object> vars)
    {
        var rows = new List<VarText>();
        foreach (var pair in vars)
            rows.Add(DescribeOne(pair.Key, pair.Value));
        return rows;
    }

    public static VarText DescribeOne(string name, object value)
    {
        switch (value)
        {
            case ImageBuffer image:
                string channels = image.Channels == 1 ? "byte" : image.Channels + " 通道";
                return Iconic(name, "image", image.Width + "×" + image.Height + "  " + channels);
            case Region region:
                return Iconic(name, "region", Objects(region.Area > 0 ? 1 : 0) + "  " + region.Width + "×" + region.Height);
            case Region[] regions:
                int width = regions.Length > 0 ? regions[0].Width : 0;
                int height = regions.Length > 0 ? regions[0].Height : 0;
                return Iconic(name, "region", Objects(regions.Length) + "  " + width + "×" + height);
            case EdgePoint[][] edges:
                return Iconic(name, "xld", edges.Length + " 條輪廓");
            case SubpixFigure figure:
                return Iconic(name, "xld", figure.Contours.Length + " 條輪廓 · " + figure.Objects.Length + " 個物件");
            case double number:
                string num = FormatNum(number);
                return Control(name, IsWhole(number) ? "integer" : "real", num, num);
            case double[] tuple:
                return Control(name, "tuple", FormatTuple(tuple, 12), FormatTuple(tuple, 400));
            case string text:
                return Control(name, "string", text, text);
            case ComplexBuffer spectrum:
                string spectrumInfo = "複數影像  " + spectrum.Width + "×" + spectrum.Height;
                return Control(name, "complex", spectrumInfo, spectrumInfo);
            default:
                string shown = value.ToString() ?? "";
                return Control(name, "tuple", shown, shown);
        }
    }

    static VarText Iconic(string name, string kind, string info) => new()
    {
        Name = name,
        Iconic = true,
        Kind = kind,
        Info = info,
        Short = info,
        Full = info,
    };

    static VarText Control(string name, string kind, string brief, string full) => new()
    {
        Name = name,
        Iconic = false,
        Kind = kind,
        Info = brief,
        Short = brief,
        Full = full,
    };

    static string Objects(int count) => count + " 個物件";

    static string FormatTuple(double[] values, int limit)
    {
        string body = string.Join(", ", values.Take(limit).Select(FormatNum));
        if (values.Length > limit)
            body += ", …";
        string text = "[" + body + "]";
        if (limit <= 12 && values.Length > 12)
            text += "  (" + values.Length + ")";
        return text;
    }

    public static string FormatNum(double value)
    {
        if (double.IsNaN(value))
            return "nan";
        if (double.IsPositiveInfinity(value))
            return "inf";
        if (double.IsNegativeInfinity(value))
            return "-inf";
        if (IsWhole(value))
            return Math.Round(value).ToString(CultureInfo.InvariantCulture);
        return value.ToString("0.####", CultureInfo.InvariantCulture);
    }

    static bool IsWhole(double value) => Math.Abs(value) < 1e15 && Math.Abs(value - Math.Round(value)) < 1e-9;
}
