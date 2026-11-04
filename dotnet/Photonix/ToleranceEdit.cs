using System.Globalization;
using System.IO;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Photonix;

sealed record ToleranceLimit(string Stem, string Label, string Min, string Max, int MinLine, int MaxLine, string MinName, string MaxName);

sealed record ToleranceExpect(string Name, string Label, string Value, int Line);

sealed record TolerancePoints(string Name, int Line);

/// <summary>和網頁版 tolerance.js 同一套：上下限、預期字、教導圓心。</summary>
static class ToleranceEdit
{
    static readonly Dictionary<string, string> Labels = new(StringComparer.Ordinal)
    {
        ["Dia"] = "直徑",
        ["Width"] = "寬",
        ["Height"] = "高",
        ["Dark"] = "暗塊",
        ["Bright"] = "亮點",
        ["Hue"] = "色調",
        ["Area"] = "面積",
        ["Count"] = "個數",
        ["Shift"] = "偏移",
    };

    static readonly Regex LimitPattern = new(@"^(\s*)([A-Za-z_]\w*?)(Min|Max)\s*:=\s*(-?\d+(?:\.\d+)?)\s*$", RegexOptions.CultureInvariant);
    static readonly Regex ExpectPattern = new(@"^(\s*)Expect\s*:=\s*'([^']*)'\s*$", RegexOptions.CultureInvariant);
    static readonly Regex PointPattern = new(@"^(\s*)(WantRow|WantCol)\s*:=\s*\[[^\]]*\]\s*$", RegexOptions.CultureInvariant);
    static readonly Regex NumberPattern = new(@"^-?\d+(?:\.\d+)?$", RegexOptions.CultureInvariant);

    public static List<ToleranceLimit> ReadLimits(string? source)
    {
        string[] lines = Lines(source);
        var found = new Dictionary<string, Slot>(StringComparer.Ordinal);
        for (int index = 0; index < lines.Length; index++)
        {
            Match match = LimitPattern.Match(lines[index]);
            if (!match.Success)
                continue;
            string stem = match.Groups[2].Value;
            if (!found.TryGetValue(stem, out Slot? slot))
            {
                slot = new Slot(stem, Labels.TryGetValue(stem, out string? label) ? label : stem);
                found[stem] = slot;
            }
            if (match.Groups[3].Value == "Min")
            {
                slot.Min = match.Groups[4].Value;
                slot.MinLine = index;
            }
            else
            {
                slot.Max = match.Groups[4].Value;
                slot.MaxLine = index;
            }
        }
        return found.Values
            .Where(slot => slot.Min != null && slot.Max != null && slot.MinLine >= 0 && slot.MaxLine >= 0)
            .OrderBy(slot => slot.MinLine)
            .Select(slot => new ToleranceLimit(slot.Stem, slot.Label, slot.Min!, slot.Max!, slot.MinLine, slot.MaxLine, slot.Stem + "Min", slot.Stem + "Max"))
            .ToList();
    }

    public static List<ToleranceExpect> ReadExpects(string? source)
    {
        string[] lines = Lines(source);
        var found = new List<ToleranceExpect>();
        for (int index = 0; index < lines.Length; index++)
        {
            Match match = ExpectPattern.Match(lines[index]);
            if (!match.Success)
                continue;
            found.Add(new ToleranceExpect("Expect", "預期字", match.Groups[2].Value, index));
        }
        return found;
    }

    public static List<TolerancePoints> ReadPointLists(string? source)
    {
        string[] lines = Lines(source);
        var found = new List<TolerancePoints>();
        for (int index = 0; index < lines.Length; index++)
        {
            Match match = PointPattern.Match(lines[index]);
            if (match.Success)
                found.Add(new TolerancePoints(match.Groups[2].Value, index));
        }
        return found;
    }

    public static string ApplyLimit(string? source, int lineIndex, string name, string value)
    {
        if (!NumberPattern.IsMatch(value))
            return source ?? "";
        return ReplaceLine(source, lineIndex, indent => indent + name + " := " + value);
    }

    public static string ApplyExpect(string? source, int lineIndex, string value)
    {
        if (value.IndexOfAny(['\'', '\r', '\n', '\\']) >= 0)
            return source ?? "";
        return ReplaceLine(source, lineIndex, indent => indent + "Expect := '" + value + "'");
    }

    public static string? ApplyPointLists(string? source, string? rowText, string? colText)
    {
        List<string>? rows = ParsePoints(rowText);
        List<string>? cols = ParsePoints(colText);
        if (rows == null || cols == null || rows.Count != cols.Count)
            return null;
        List<TolerancePoints> lists = ReadPointLists(source);
        TolerancePoints? rowLine = lists.FirstOrDefault(item => item.Name == "WantRow");
        TolerancePoints? colLine = lists.FirstOrDefault(item => item.Name == "WantCol");
        if (rowLine == null || colLine == null)
            return null;
        string[] lines = Lines(source);
        WritePoints(lines, rowLine, "WantRow", rows);
        WritePoints(lines, colLine, "WantCol", cols);
        return string.Join("\n", lines);
    }

    public static string? PointText(object? value)
    {
        if (value is not double[] many || many.Length == 0 || many.Length > 64)
            return null;
        var parts = new List<string>();
        foreach (double number in many)
        {
            if (double.IsNaN(number) || double.IsInfinity(number))
                return null;
            double rounded = Math.Round(number * 100, MidpointRounding.ToPositiveInfinity) / 100;
            if (rounded == 0)
                rounded = 0;
            parts.Add(rounded.ToString("0.##", CultureInfo.InvariantCulture));
        }
        return "[" + string.Join(", ", parts) + "]";
    }

    public static void Remember(string? fileName, string? source)
    {
        string key = FileKey(fileName);
        if (key.Length == 0)
            return;
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (ToleranceLimit field in ReadLimits(source))
        {
            values[field.MinName] = field.Min;
            values[field.MaxName] = field.Max;
        }
        foreach (ToleranceExpect field in ReadExpects(source))
            values[field.Name] = field.Value;
        if (values.Count == 0)
            return;
        Dictionary<string, Dictionary<string, string>> all = ReadStore();
        all[key] = values;
        WriteStore(all);
    }

    public static string Restore(string? fileName, string? source)
    {
        string text = source ?? "";
        string key = FileKey(fileName);
        if (key.Length == 0 || !ReadStore().TryGetValue(key, out Dictionary<string, string>? saved) || saved == null)
            return text;
        string next = text;
        foreach (ToleranceLimit field in ReadLimits(next))
        {
            if (saved.TryGetValue(field.MinName, out string? min))
                next = ApplyLimit(next, field.MinLine, field.MinName, min);
            if (saved.TryGetValue(field.MaxName, out string? max))
                next = ApplyLimit(next, ReadLimits(next).First(item => item.Stem == field.Stem).MaxLine, field.MaxName, max);
        }
        foreach (ToleranceExpect field in ReadExpects(next))
        {
            if (saved.TryGetValue(field.Name, out string? value))
                next = ApplyExpect(next, field.Line, value);
        }
        return next;
    }

    public static void Clear(string? fileName)
    {
        string key = FileKey(fileName);
        if (key.Length == 0)
            return;
        Dictionary<string, Dictionary<string, string>> all = ReadStore();
        if (!all.Remove(key))
            return;
        WriteStore(all);
    }

    static void WritePoints(string[] lines, TolerancePoints slot, string name, List<string> numbers)
    {
        string indent = Indent(lines[slot.Line]);
        lines[slot.Line] = indent + name + " := [" + string.Join(", ", numbers) + "]";
    }

    static List<string>? ParsePoints(string? text)
    {
        string raw = text ?? "";
        Match match = Regex.Match(raw, @"\[([^\]]*)\]", RegexOptions.CultureInvariant);
        if (!match.Success || match.Groups[1].Value.Contains('…'))
            return null;
        string[] parts = match.Groups[1].Value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length == 0 || parts.Length > 64)
            return null;
        var numbers = new List<string>();
        foreach (string part in parts)
        {
            if (!NumberPattern.IsMatch(part) || !double.TryParse(part, NumberStyles.Float, CultureInfo.InvariantCulture, out double number) || !double.IsFinite(number))
                return null;
            double rounded = Math.Round(number * 100, MidpointRounding.ToPositiveInfinity) / 100;
            if (rounded == 0)
                rounded = 0;
            numbers.Add(rounded.ToString("0.##", CultureInfo.InvariantCulture));
        }
        return numbers;
    }

    static string ReplaceLine(string? source, int lineIndex, Func<string, string> write)
    {
        string[] lines = Lines(source);
        if (lineIndex < 0 || lineIndex >= lines.Length)
            return source ?? "";
        lines[lineIndex] = write(Indent(lines[lineIndex]));
        return string.Join("\n", lines);
    }

    static string Indent(string line)
    {
        int count = 0;
        while (count < line.Length && line[count] == ' ')
            count++;
        return line[..count];
    }

    static string[] Lines(string? source) => (source ?? "").Replace("\r\n", "\n").Replace('\r', '\n').Split('\n');

    static string FileKey(string? name) => Path.GetFileName((name ?? "").Replace('\\', '/').Trim());

    static string StorePath() => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Photonix",
        "tolerances.json");

    static Dictionary<string, Dictionary<string, string>> ReadStore()
    {
        try
        {
            string path = StorePath();
            if (!File.Exists(path))
                return new Dictionary<string, Dictionary<string, string>>(StringComparer.Ordinal);
            return JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, string>>>(File.ReadAllText(path))
                ?? new Dictionary<string, Dictionary<string, string>>(StringComparer.Ordinal);
        }
        catch (Exception)
        {
            return new Dictionary<string, Dictionary<string, string>>(StringComparer.Ordinal);
        }
    }

    static void WriteStore(Dictionary<string, Dictionary<string, string>> data)
    {
        string path = StorePath();
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, JsonSerializer.Serialize(data));
    }

    sealed class Slot
    {
        public Slot(string stem, string label)
        {
            Stem = stem;
            Label = label;
        }

        public string Stem { get; }
        public string Label { get; }
        public string? Min { get; set; }
        public string? Max { get; set; }
        public int MinLine { get; set; } = -1;
        public int MaxLine { get; set; } = -1;
    }
}
