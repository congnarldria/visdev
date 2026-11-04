using System.Text.RegularExpressions;

namespace Photonix;

/// <summary>和網頁版次像素視窗同一套：把門檻寫進 threshold_sub_pix。</summary>
static class SubpixProgram
{
    static readonly Regex BinaryLine = new(@"^\s*(?:Vizion\.)?(threshold|opencv_threshold|binary_threshold)\s*\(", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
    static readonly Regex SubpixLine = new(@"^\s*(?:Vizion\.)?threshold_sub_pix\s*\(", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
    static readonly Regex NumberText = new(@"^-?\d+(?:\.\d+)?$", RegexOptions.CultureInvariant);
    static readonly Regex DisplayLine = new(@"^\s*(?:Vizion\.)?dev_display\s*\(", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

    public readonly record struct State(int Level, string Image, string Region, string Contour);

    public static State Read(string? source)
    {
        var lines = Lines(source);
        string image = GuessImage(lines);
        string region = "Region";
        string contour = "Border";
        int level = 128;
        foreach (string line in lines)
        {
            var binary = Regex.Match(line, @"^\s*(?:Vizion\.)?(?:threshold|opencv_threshold|binary_threshold)\s*\((.*)\)\s*$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            if (binary.Success)
            {
                string[] args = SplitArgs(binary.Groups[1].Value).Select(item => item.Trim()).ToArray();
                if (args.Length > 1 && args[1].Length > 0)
                    region = args[1];
                if (args.Length > 2 && NumberText.IsMatch(args[2]))
                    level = Clamp(args[2]);
            }
            var sub = Regex.Match(line, @"^\s*(?:Vizion\.)?threshold_sub_pix\s*\((.*)\)\s*$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            if (!sub.Success)
                continue;
            string[] subArgs = SplitArgs(sub.Groups[1].Value).Select(item => item.Trim()).ToArray();
            if (subArgs.Length > 0 && subArgs[0].Length > 0)
                image = subArgs[0];
            if (subArgs.Length > 1 && subArgs[1].Length > 0)
                contour = subArgs[1];
            if (subArgs.Length > 2 && NumberText.IsMatch(subArgs[2]))
                level = Clamp(subArgs[2]);
        }
        return new State(level, image, region, contour);
    }

    public static string Apply(string? source, int level)
    {
        int value = Math.Clamp(level, 0, 255);
        var lines = Lines(source);
        string image = GuessImage(lines);
        string region = "Region";
        int binaryAt = lines.FindIndex(line => BinaryLine.IsMatch(line));
        if (binaryAt >= 0)
        {
            var match = Regex.Match(lines[binaryAt], @"^\s*(?:Vizion\.)?(?:threshold|opencv_threshold|binary_threshold)\s*\((.*)\)\s*$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            if (match.Success)
            {
                string[] args = SplitArgs(match.Groups[1].Value).Select(item => item.Trim()).ToArray();
                if (args.Length > 1 && args[1].Length > 0)
                    region = args[1];
            }
        }
        else
        {
            int displayAt = lines.FindIndex(line => DisplayLine.IsMatch(line));
            lines.Insert(displayAt >= 0 ? displayAt : lines.Count, Call("threshold", $"{image}, {region}, {value}, 255"));
        }

        string contour = "Border";
        int index = lines.FindIndex(line => SubpixLine.IsMatch(line));
        if (index >= 0)
        {
            var match = Regex.Match(lines[index], @"^(\s*)(?:Vizion\.)?threshold_sub_pix\s*\((.*)\)\s*$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            if (match.Success)
            {
                string[] args = SplitArgs(match.Groups[2].Value).Select(item => item.Trim()).ToArray();
                if (args.Length > 0 && args[0].Length > 0)
                    image = args[0];
                if (args.Length > 1 && args[1].Length > 0)
                    contour = args[1];
                lines[index] = match.Groups[1].Value + Call("threshold_sub_pix", $"{image}, {contour}, {value}");
            }
        }
        else
        {
            int displayAt = lines.FindIndex(line => DisplayLine.IsMatch(line));
            lines.Insert(displayAt >= 0 ? displayAt : lines.Count, Call("threshold_sub_pix", $"{image}, {contour}, {value}"));
        }
        if (!Shown(lines, region))
            AddLine(lines, Call("dev_display", region));
        if (!Shown(lines, contour))
            AddLine(lines, Call("dev_display", contour));
        return string.Join("\n", lines);
    }

    static void AddLine(List<string> lines, string call)
    {
        int at = lines.Count;
        while (at > 0 && string.IsNullOrWhiteSpace(lines[at - 1]))
            at--;
        lines.Insert(at, call);
    }

    static bool Shown(List<string> lines, string name)
    {
        string pattern = @"^\s*(?:Vizion\.)?dev_display\s*\(\s*" + Regex.Escape(name) + @"\s*\)\s*$";
        return lines.Exists(line => Regex.IsMatch(line, pattern, RegexOptions.IgnoreCase | RegexOptions.CultureInvariant));
    }

    static int Clamp(string text)
    {
        double number = double.Parse(text, System.Globalization.CultureInfo.InvariantCulture);
        return Math.Clamp((int)Math.Round(number), 0, 255);
    }

    static string Call(string name, string args) => "Vizion." + name + " (" + args + ")";

    static List<string> Lines(string? source) => (source ?? "").Replace("\r\n", "\n").Replace('\r', '\n').Split('\n').ToList();

    static string GuessImage(List<string> lines)
    {
        string name = "Image";
        foreach (string line in lines)
        {
            var match = Regex.Match(line, @"^\s*(?:Vizion\.)?(read_image|rgb1_to_gray|gauss_image|median_image|mean_image|emphasize|scale_image|invert_image)\s*\((.*)\)\s*$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            if (!match.Success)
                continue;
            List<string> args = SplitArgs(match.Groups[2].Value);
            string? picked = match.Groups[1].Value.Equals("read_image", StringComparison.OrdinalIgnoreCase) ? args.ElementAtOrDefault(0) : args.ElementAtOrDefault(1);
            if (!string.IsNullOrWhiteSpace(picked))
                name = picked.Trim();
        }
        return name;
    }

    static List<string> SplitArgs(string inner)
    {
        var parts = new List<string>();
        var current = new System.Text.StringBuilder();
        char quote = '\0';
        foreach (char ch in inner)
        {
            if (quote != '\0')
            {
                current.Append(ch);
                if (ch == quote)
                    quote = '\0';
                continue;
            }
            if (ch is '\'' or '"')
            {
                quote = ch;
                current.Append(ch);
                continue;
            }
            if (ch == ',')
            {
                parts.Add(current.ToString());
                current.Clear();
                continue;
            }
            current.Append(ch);
        }
        if (current.Length > 0 || parts.Count > 0)
            parts.Add(current.ToString());
        return parts;
    }
}
