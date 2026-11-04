using System.Globalization;
using System.Text.RegularExpressions;

namespace Photonix;

/// <summary>圖形列的 OCR 與 connection，和網頁版同一套改程式。</summary>
static class GraphicsTools
{
    public readonly record struct Engine(string Key, string Title, string Tip);

    public static readonly Engine[] Engines =
    {
        new("auto", "自動", "在已安裝的本機通用引擎裡挑。不把影像送到雲端。"),
        new("rapid", "RapidOCR", "本機通用。PP-OCR 的 ONNX 版，面向中文與英文。"),
        new("paddle", "PaddleOCR", "本機通用。百度 PP-OCR，適合歪斜與拍照中文。"),
        new("easyocr", "EasyOCR", "本機通用。多語場景文字，依賴 PyTorch。"),
        new("cnocr", "CnOCR", "本機通用。中文印刷體。"),
        new("windows", "Windows OCR", "本機通用。用系統語言包，不另下載模型。"),
        new("tesseract", "Tesseract", "本機通用。適合乾淨掃描件。繁體要 chi_tra。"),
        new("doctr", "docTR", "文件。適合拉丁文掃描件與表單。"),
        new("surya", "Surya", "文件。適合多欄與部分手寫。"),
        new("mmocr", "MMOCR", "文件。研究用的偵測與識別。"),
        new("kerasocr", "Keras OCR", "文件。英文場景文字的對照。"),
        new("trocr", "TrOCR", "文件。單行印刷或手寫，整頁會被當成一行。"),
        new("got", "GOT-OCR 2.0", "文件。整頁轉成文字，不逐行給外框。"),
        new("manga", "MangaOCR", "文件。日文漫畫對白，要先裁好對話框。"),
        new("ddddocr", "ddddocr", "文件。驗證碼與簡單字元，不適合長文。"),
        new("azure", "Azure AI Vision", "雲端。影像會離開這台電腦。"),
        new("google", "Google Cloud Vision", "雲端。影像會上傳到 Google。"),
        new("textract", "Amazon Textract", "雲端。適合表單與掃描件。"),
        new("baidu", "百度文字識別", "雲端。中文印刷與拍照。"),
        new("tencent", "騰訊雲 OCR", "雲端。中文票據與文件。"),
        new("aliyun", "阿里雲 OCR", "雲端。中文文件與卡證。"),
        new("huawei", "華為雲 OCR", "雲端。中文文件與表格。"),
        new("xfyun", "訊飛 OCR", "雲端。中英印刷與拍照。"),
        new("youdao", "有道 OCR", "雲端。中英文件與自然場景。"),
        new("mathpix", "Mathpix", "雲端。數學與化學式，不是一般場景文字。"),
        new("ocrspace", "OCR.space", "雲端。少量多語文件的備援。"),
        new("abbyy", "ABBYY Cloud OCR", "雲端。發票、表單與掃描件。"),
    };

    static readonly Regex BinaryLine = new(@"^\s*(?:Vizion\.)?(threshold|opencv_threshold|binary_threshold)\s*\(", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
    public static string ApplyOcr(string? source, string? method)
    {
        string key = string.IsNullOrWhiteSpace(method) ? "auto" : method.Trim();
        var lines = Lines(source);
        string image = GuessImage(lines);
        string call = Call("ocr_find", image + ", Words, '" + key + "', Text, Row, Column, Score");
        int index = lines.FindIndex(line => Regex.IsMatch(line, @"^\s*(?:Vizion\.)?ocr_find\s*\(", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant));
        if (index >= 0)
            lines[index] = call;
        else
            AddLine(lines, call);
        return string.Join("\n", lines);
    }

    public static string? ReadOcrMethod(string? source)
    {
        var match = Regex.Match(source ?? "", @"ocr_find\s*\(\s*[^,]+,\s*[^,]+,\s*'([^']+)'", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        return match.Success ? match.Groups[1].Value : null;
    }

    public static bool TryDistance(string? source, out double distance)
    {
        distance = 1;
        string text = source ?? "";
        var opencv = Regex.Match(text, @"opencv_connection\s*\(\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*([0-9.]+)\s*\)", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        var halcon = Regex.Match(text, @"(?<!opencv_)connection\s*\(\s*[^,]+,\s*[^,)]+\s*,\s*([0-9.]+)\s*\)", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        string raw = opencv.Success ? opencv.Groups[1].Value : halcon.Success ? halcon.Groups[1].Value : "";
        return raw.Length > 0 && double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out distance);
    }

    public static bool TryMinArea(string? source, out int area)
    {
        area = 0;
        var match = Regex.Match(source ?? "", @"select_shape\s*\([^)]*'area'[^)]*,\s*([0-9.]+)\s*,", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        if (!match.Success || !double.TryParse(match.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out double value))
            return false;
        area = Math.Max(0, (int)Math.Round(value, MidpointRounding.AwayFromZero));
        return true;
    }

    public static int ParseArea(string? text)
    {
        if (!double.TryParse((text ?? "").Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double value) || double.IsNaN(value) || double.IsInfinity(value))
            return 0;
        return Math.Max(0, (int)Math.Round(value, MidpointRounding.AwayFromZero));
    }

    public static double ParseDistance(string? text)
    {
        if (!double.TryParse((text ?? "").Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double value) || double.IsNaN(value) || double.IsInfinity(value) || value < 1)
            return 1;
        return Math.Round(value * 1000, MidpointRounding.AwayFromZero) / 1000;
    }

    public static string FormatDistance(double value)
    {
        double rounded = Math.Round(value, 3, MidpointRounding.AwayFromZero);
        if (Math.Abs(rounded - Math.Round(rounded)) < 1e-9)
            return Math.Round(rounded).ToString(CultureInfo.InvariantCulture);
        return rounded.ToString("0.###", CultureInfo.InvariantCulture);
    }

    public static string ApplyConnection(string? source, int minArea, double distance)
    {
        int area = Math.Max(0, minArea);
        double gap = distance < 1 ? 1 : Math.Round(distance * 1000, MidpointRounding.AwayFromZero) / 1000;
        string gapText = FormatDistance(gap);
        var lines = Lines(source);
        string region = "Region";
        bool opencv = false;
        int binaryAt = lines.FindIndex(line => BinaryLine.IsMatch(line));
        if (binaryAt >= 0)
        {
            var inner = Regex.Match(lines[binaryAt], @"\((.*)\)");
            if (inner.Success)
            {
                string[] args = SplitArgs(inner.Groups[1].Value).Select(item => item.Trim()).ToArray();
                if (args.Length > 1 && args[1].Length > 0)
                    region = args[1];
            }
            opencv = Regex.IsMatch(lines[binaryAt], @"(?:Vizion\.)?opencv_threshold", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        }
        string regionPattern = Regex.Escape(region);
        int connIndex = lines.FindIndex(line => Regex.IsMatch(line, @"^\s*(?:Vizion\.)?(connection|opencv_connection)\s*\(\s*" + regionPattern + @"\s*,", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant));
        string output = "Connected";
        if (connIndex >= 0)
        {
            var match = Regex.Match(lines[connIndex], @"^(\s*)(?:Vizion\.)?(connection|opencv_connection)\s*\((.*)\)\s*$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            if (match.Success)
            {
                string[] args = SplitArgs(match.Groups[3].Value).Select(item => item.Trim()).ToArray();
                if (args.Length > 1 && args[1].Length > 0)
                    output = args[1];
                if (match.Groups[2].Value.Equals("opencv_connection", StringComparison.OrdinalIgnoreCase))
                {
                    string linked = args.Length > 2 && args[2].Length > 0 ? args[2] : "8";
                    string image = args.Length > 0 ? args[0] : region;
                    lines[connIndex] = match.Groups[1].Value + Call("opencv_connection", image + ", " + output + ", " + linked + ", " + gapText);
                }
                else
                {
                    string image = args.Length > 0 ? args[0] : region;
                    lines[connIndex] = match.Groups[1].Value + Call("connection", image + ", " + output + ", " + gapText);
                }
            }
        }
        else
        {
            int insertAt = binaryAt >= 0 ? binaryAt + 1 : lines.Count;
            if (binaryAt < 0)
            {
                while (insertAt > 0 && string.IsNullOrWhiteSpace(lines[insertAt - 1]))
                    insertAt--;
            }
            string call = opencv
                ? Call("opencv_connection", region + ", " + output + ", 8, " + gapText)
                : Call("connection", region + ", " + output + ", " + gapText);
            lines.Insert(insertAt, call);
            connIndex = insertAt;
        }
        string outputPattern = Regex.Escape(output);
        int shapeIndex = -1;
        for (int i = connIndex + 1; i < lines.Count; i++)
        {
            if (Regex.IsMatch(lines[i], @"^\s*(?:Vizion\.)?select_shape\s*\(\s*" + outputPattern + @"\s*,", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)
                && Regex.IsMatch(lines[i], "area", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant))
            {
                shapeIndex = i;
                break;
            }
        }
        string shown = output;
        if (shapeIndex >= 0)
        {
            var match = Regex.Match(lines[shapeIndex], @"^(\s*)(?:Vizion\.)?select_shape\s*\((.*)\)\s*$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            if (match.Success)
            {
                var args = SplitArgs(match.Groups[2].Value);
                if (args.Count >= 6)
                {
                    args[4] = " " + area;
                    lines[shapeIndex] = match.Groups[1].Value + Call("select_shape", string.Join(", ", args.Select(item => item.Trim())));
                }
                if (args.Count > 1 && args[1].Trim().Length > 0)
                    shown = args[1].Trim();
            }
        }
        else
        {
            shown = "Selected";
            lines.Insert(connIndex + 1, Call("select_shape", output + ", Selected, 'area', 'and', " + area + ", 99999999"));
        }
        if (!lines.Exists(line => Regex.IsMatch(line, @"^\s*(?:Vizion\.)?dev_display\s*\(\s*" + Regex.Escape(shown) + @"\s*\)\s*$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)))
        {
            int displayIndex = lines.FindIndex(line => Regex.IsMatch(line, @"^\s*(?:Vizion\.)?dev_display\s*\(\s*" + regionPattern + @"\s*\)\s*$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant));
            if (displayIndex >= 0)
            {
                string indent = Regex.Match(lines[displayIndex], @"^\s*").Value;
                lines[displayIndex] = indent + Call("dev_display", shown);
            }
            else
            {
                AddLine(lines, Call("dev_set_draw", "'fill'"));
                AddLine(lines, Call("dev_display", shown));
            }
        }
        return string.Join("\n", lines);
    }

    static void AddLine(List<string> lines, string call)
    {
        int at = lines.Count;
        while (at > 0 && string.IsNullOrWhiteSpace(lines[at - 1]))
            at--;
        lines.Insert(at, call);
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
