using System.Text.RegularExpressions;

namespace Photonix;

/// <summary>和網頁版二值化視窗同一套：改程式裡的門檻呼叫。</summary>
static class ThresholdProgram
{
    static readonly Regex BinaryLine = new(@"^\s*(?:Vizion\.)?(threshold|opencv_threshold|binary_threshold)\s*\(", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
    static readonly Regex NumberText = new(@"^-?\d+(?:\.\d+)?$", RegexOptions.CultureInvariant);

    public readonly record struct State(string Kind, string LightDark, string Method, int Line, bool Literal, int Min, int Max, int Level);

    public static State Read(string? source)
    {
        string[] lines = Lines(source);
        for (int index = 0; index < lines.Length; index++)
        {
            if (!BinaryLine.IsMatch(lines[index]))
                continue;
            var match = Regex.Match(lines[index], @"^\s*(?:Vizion\.)?(threshold|opencv_threshold|binary_threshold)\s*\((.*)\)\s*$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            if (!match.Success)
                continue;
            string name = match.Groups[1].Value.ToLowerInvariant();
            string[] args = SplitArgs(match.Groups[2].Value).Select(item => item.Trim()).ToArray();
            int level = 128;
            int max = 255;
            bool literal = false;
            string light = "light";
            string method = "max_separability";
            if (name == "threshold")
            {
                literal = args.Length >= 4 && NumberText.IsMatch(args[^2]) && NumberText.IsMatch(args[^1]);
                if (args.Length >= 3 && NumberText.IsMatch(args[2]))
                    level = (int)Math.Round(double.Parse(args[2], System.Globalization.CultureInfo.InvariantCulture));
                if (args.Length >= 4 && NumberText.IsMatch(args[3]))
                    max = (int)Math.Round(double.Parse(args[3], System.Globalization.CultureInfo.InvariantCulture));
                return new State("range", "light", "threshold", index, literal, level, max, level);
            }
            if (name == "opencv_threshold")
            {
                literal = args.Length >= 3 && NumberText.IsMatch(args[2]);
                if (literal)
                    level = (int)Math.Round(double.Parse(args[2], System.Globalization.CultureInfo.InvariantCulture));
                string mode = args.Length >= 4 ? args[3].Trim('\'', '"') : "binary";
                return new State("opencv", mode, "opencv", index, literal, 0, mode == "binary_inv" ? level : 255, level);
            }
            method = args.Length >= 3 ? args[2].Trim('\'', '"') : method;
            light = args.Length >= 4 ? args[3].Trim('\'', '"') : light;
            string kind = method is "local_mean" or "local_gauss" ? "local" : method == "max_separability" ? "otsu" : "auto";
            return new State(kind, light, method, index, false, 0, 255, level);
        }
        return new State("range", "light", "threshold", -1, true, 128, 255, 128);
    }

    public static string ApplyTool(string? source, string tool, int level, int maxGray)
    {
        var lines = Lines(source).ToList();
        int index = lines.FindIndex(line => BinaryLine.IsMatch(line));
        string image = GuessImage(lines);
        string region = "Region";
        string used = "UsedThreshold";
        string indent = "";
        if (index >= 0)
        {
            var match = Regex.Match(lines[index], @"^(\s*)(?:Vizion\.)?(?:threshold|opencv_threshold|binary_threshold)\s*\((.*)\)\s*$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            if (match.Success)
            {
                indent = match.Groups[1].Value;
                string[] args = SplitArgs(match.Groups[2].Value).Select(item => item.Trim()).ToArray();
                string name = Regex.Match(lines[index], @"opencv_threshold|binary_threshold|threshold", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant).Value.ToLowerInvariant();
                if (args.Length > 0 && args[0].Length > 0)
                    image = args[0];
                if (args.Length > 1 && args[1].Length > 0)
                    region = args[1];
                if (name == "threshold")
                {
                    if (args.Length > 2 && NumberText.IsMatch(args[2]))
                        level = (int)Math.Round(double.Parse(args[2], System.Globalization.CultureInfo.InvariantCulture));
                    if (args.Length > 3 && NumberText.IsMatch(args[3]))
                        maxGray = (int)Math.Round(double.Parse(args[3], System.Globalization.CultureInfo.InvariantCulture));
                }
                else if (name == "opencv_threshold" && args.Length > 2 && NumberText.IsMatch(args[2]))
                    level = (int)Math.Round(double.Parse(args[2], System.Globalization.CultureInfo.InvariantCulture));
                else if (name == "binary_threshold" && args.Length > 4 && args[4].Length > 0)
                    used = args[4];
            }
        }
        string call = Call("binary_threshold", $"{image}, {region}, 'max_separability', 'light', {used}");
        if (tool == "halcon")
            call = Call("threshold", $"{image}, {region}, {level}, {maxGray}");
        else if (tool == "opencv")
            call = Call("opencv_threshold", $"{image}, {region}, {level}, 'binary'");
        else if (tool == "inv")
            call = Call("opencv_threshold", $"{image}, {region}, {level}, 'binary_inv'");
        else if (tool == "dark")
            call = Call("binary_threshold", $"{image}, {region}, 'max_separability', 'dark', {used}");
        else if (tool == "smooth")
            call = Call("binary_threshold", $"{image}, {region}, 'smooth_histo', 'light', {used}");
        else if (tool == "triangle")
            call = Call("binary_threshold", $"{image}, {region}, 'triangle', 'light', {used}");
        else if (tool == "local")
            call = Call("binary_threshold", $"{image}, {region}, 'local_mean', 'light', {used}");
        else if (tool == "lgauss")
            call = Call("binary_threshold", $"{image}, {region}, 'local_gauss', 'light', {used}");
        if (index >= 0)
            lines[index] = indent + call;
        else
        {
            int displayAt = lines.FindIndex(line => Regex.IsMatch(line, @"^\s*(?:Vizion\.)?dev_display\s*\(", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant));
            lines.Insert(displayAt >= 0 ? displayAt : lines.Count, call);
        }
        bool opencv = tool is "opencv" or "inv";
        for (int lineIndex = 0; lineIndex < lines.Count; lineIndex++)
        {
            string line = lines[lineIndex];
            if (opencv)
            {
                var linked = Regex.Match(line, @"^(\s*)(?:Vizion\.)?connection\s*\(\s*([^,]+)\s*,\s*([^,)]+)\s*(?:,\s*([^)]+))?\s*\)\s*$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
                if (linked.Success && linked.Groups[2].Value.Trim() == region)
                {
                    string dist = linked.Groups[4].Success && linked.Groups[4].Value.Trim().Length > 0 ? linked.Groups[4].Value.Trim() : "1";
                    lines[lineIndex] = linked.Groups[1].Value + Call("opencv_connection", $"{linked.Groups[2].Value.Trim()}, {linked.Groups[3].Value.Trim()}, 8, {dist}");
                }
            }
            else
            {
                var linked = Regex.Match(line, @"^(\s*)(?:Vizion\.)?opencv_connection\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,)]+)\s*(?:,\s*([^)]+))?\s*\)\s*$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
                if (linked.Success && linked.Groups[2].Value.Trim() == region)
                {
                    string dist = linked.Groups[5].Success && linked.Groups[5].Value.Trim().Length > 0 ? linked.Groups[5].Value.Trim() : "1";
                    lines[lineIndex] = linked.Groups[1].Value + Call("connection", $"{linked.Groups[2].Value.Trim()}, {linked.Groups[3].Value.Trim()}, {dist}");
                }
            }
        }
        return string.Join("\n", lines);
    }

    public static string? RewriteRange(string? source, int line, int minGray, int maxGray)
    {
        var lines = Lines(source).ToList();
        if (line < 0 || line >= lines.Count)
            return null;
        var match = Regex.Match(lines[line], @"^(\s*)(?:Vizion\.)?threshold\s*\((.*)\)\s*$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        if (!match.Success)
            return null;
        var args = SplitArgs(match.Groups[2].Value);
        if (args.Count < 2)
            return null;
        if (!NumberText.IsMatch(args[^1].Trim()) || !NumberText.IsMatch(args[^2].Trim()))
            return null;
        args[^2] = " " + minGray;
        args[^1] = " " + maxGray;
        lines[line] = match.Groups[1].Value + Call("threshold", string.Join(",", args));
        return string.Join("\n", lines);
    }

    public static string? RewriteLevel(string? source, int line, int level)
    {
        var lines = Lines(source).ToList();
        if (line < 0 || line >= lines.Count)
            return null;
        var match = Regex.Match(lines[line], @"^(\s*)(?:Vizion\.)?opencv_threshold\s*\((.*)\)\s*$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        if (!match.Success)
            return null;
        var args = SplitArgs(match.Groups[2].Value);
        if (args.Count < 3 || !NumberText.IsMatch(args[2].Trim()))
            return null;
        args[2] = " " + level;
        lines[line] = match.Groups[1].Value + Call("opencv_threshold", string.Join(",", args));
        return string.Join("\n", lines);
    }

    public static int SumBins(int[] bins, int minGray, int maxGray)
    {
        int total = 0;
        int start = Math.Max(0, minGray);
        int end = Math.Min(255, maxGray);
        for (int i = start; i <= end; i++)
            total += bins[i];
        return total;
    }

    static string Call(string name, string args) => "Vizion." + name + " (" + args + ")";

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

    static string[] Lines(string? source) => (source ?? "").Replace("\r\n", "\n").Replace('\r', '\n').Split('\n');
}
