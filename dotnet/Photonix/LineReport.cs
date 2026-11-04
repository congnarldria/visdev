using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;

namespace Photonix;

/// <summary>產線明細。只讀 workspace 裡的 CSV，和不合格圖資料夾裡的檔。</summary>
static class LineReport
{
    public readonly record struct Stats(string File, int Total, int Ok, int Ng, string Yield, bool NoJudge, string LastNg);

    public readonly record struct Sheet(string File, int Total, int NgTotal, string Header, string[] Rows, string[] Opens, bool Truncated, bool OnlyNg, bool NoJudge);

    static readonly HashSet<string> Suffixes = new(StringComparer.OrdinalIgnoreCase)
    {
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp",
    };

    public static string? CsvFile(string? source, IReadOnlyDictionary<string, object> vars)
    {
        if (vars.TryGetValue("CsvFile", out object? value) && value is string text)
            return text.Trim().Trim('\'', '"');
        return Literal(source, "CsvFile");
    }

    public static string? LastNg(string? source, IReadOnlyDictionary<string, object> vars)
    {
        if (vars.TryGetValue("LastNg", out object? value) && value is string text)
        {
            string name = text.Trim().Trim('\'', '"');
            return name.Length == 0 ? null : name;
        }
        string? literal = Literal(source, "LastNg");
        return string.IsNullOrEmpty(literal) ? null : literal;
    }

    public static Stats ReadStats(string repo, string csvFile)
    {
        (string shown, string[] lines) = ReadLines(repo, csvFile);
        string[] body = lines.Skip(1).ToArray();
        int? judgeAt = Column(lines[0], "judge", "判定");
        int? fileAt = Column(lines[0], "file", "檔名");
        if (judgeAt == null)
            return new Stats(shown, body.Length, 0, 0, "", true, "");
        int ok = 0;
        int ng = 0;
        string last = "";
        foreach (string row in body)
        {
            if (IsNg(row, judgeAt))
            {
                ng++;
                string name = OpenName(row, fileAt, judgeAt);
                if (name.Length > 0)
                    last = name;
            }
            else if (IsOk(row, judgeAt))
                ok++;
        }
        return new Stats(shown, body.Length, ok, ng, Yield(ok, body.Length), false, last);
    }

    public static Sheet ReadSheet(string repo, string csvFile, bool onlyNg, int limit = 20)
    {
        (string shown, string[] lines) = ReadLines(repo, csvFile);
        string[] body = lines.Skip(1).ToArray();
        int? judgeAt = Column(lines[0], "judge", "判定");
        int? fileAt = Column(lines[0], "file", "檔名");
        string[] ngRows = body.Where(row => IsNg(row, judgeAt)).ToArray();
        string[] chosen = onlyNg ? ngRows : body;
        bool truncated = chosen.Length > limit;
        string[] shownRows = truncated ? chosen.Skip(chosen.Length - limit).ToArray() : chosen;
        return new Sheet(
            shown,
            body.Length,
            ngRows.Length,
            lines[0],
            shownRows,
            shownRows.Select(row => OpenName(row, fileAt, judgeAt)).ToArray(),
            truncated,
            onlyNg,
            judgeAt == null);
    }

    public static string Note(Sheet sheet)
    {
        if (sheet.NoJudge)
            return sheet.File + "，這份明細沒有判定欄";
        if (sheet.OnlyNg && sheet.NgTotal == 0)
            return sheet.File + "，共 " + sheet.Total + " 列，沒有不合格的列";
        if (sheet.OnlyNg && sheet.Truncated)
            return sheet.File + "，共 " + sheet.Total + " 列，不合格 " + sheet.NgTotal + " 列，這裡是最後 " + sheet.Rows.Length + " 列不合格";
        if (sheet.OnlyNg)
            return sheet.File + "，共 " + sheet.Total + " 列，不合格 " + sheet.NgTotal + " 列";
        if (sheet.Truncated)
            return sheet.File + "，共 " + sheet.Total + " 列，不合格 " + sheet.NgTotal + " 列，這裡是最後 " + sheet.Rows.Length + " 列";
        return sheet.File + "，共 " + sheet.Total + " 列，不合格 " + sheet.NgTotal + " 列";
    }

    public static string NgPath(string repo, string name)
    {
        string filename = NgFilename(name);
        string folder = Path.GetFullPath(Path.Combine(repo, "workspace", "line_ng"));
        string path = Path.GetFullPath(Path.Combine(folder, filename));
        if (!path.StartsWith(folder + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase) && !path.Equals(folder, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("不合格圖不能離開不合格圖資料夾");
        if (!File.Exists(path))
            throw new FileNotFoundException("找不到不合格圖 " + filename + "。它可能已被清掉，請再跑一次產線。");
        return path;
    }

    static (string Shown, string[] Lines) ReadLines(string repo, string csvFile)
    {
        string text = (csvFile ?? "").Trim().Trim('\'', '"');
        if (text.Length == 0)
            throw new InvalidOperationException("這一輪沒有明細");
        string normalized = text.Replace('\\', '/');
        if (Path.IsPathRooted(text) || normalized.StartsWith('/') || normalized.Split('/').Contains(".."))
            throw new InvalidOperationException("明細路徑不正確");
        string workspace = Path.GetFullPath(Path.Combine(repo, "workspace"));
        string path = Path.GetFullPath(Path.Combine(repo, normalized.Replace('/', Path.DirectorySeparatorChar)));
        if (!path.StartsWith(workspace + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("明細只能讀 workspace 裡的 CSV");
        if (!path.EndsWith(".csv", StringComparison.OrdinalIgnoreCase) || !File.Exists(path))
            throw new FileNotFoundException("找不到明細 " + normalized);
        string[] lines = File.ReadAllLines(path);
        if (lines.Length == 0)
            throw new InvalidOperationException("明細是空的");
        return (normalized, lines);
    }

    static string? Literal(string? source, string name)
    {
        string? found = null;
        foreach (Match match in Regex.Matches(source ?? "", @"^\s*" + Regex.Escape(name) + @"\s*:=\s*'([^']*)'\s*$", RegexOptions.Multiline | RegexOptions.CultureInvariant))
            found = match.Groups[1].Value;
        return found;
    }

    static string[] Cells(string row) => row.Split(',').Select(cell => cell.Trim()).ToArray();

    static int? Column(string header, params string[] names)
    {
        string[] cells = Cells(header).Select(cell => cell.ToLowerInvariant()).ToArray();
        foreach (string name in names)
        {
            int index = Array.IndexOf(cells, name.ToLowerInvariant());
            if (index >= 0)
                return index;
        }
        return null;
    }

    static string Token(string row, int? judgeAt)
    {
        if (judgeAt == null)
            return "";
        string[] cells = Cells(row);
        if (judgeAt.Value >= cells.Length)
            return "";
        return cells[judgeAt.Value].Trim().Trim('\'', '"');
    }

    static bool IsNg(string row, int? judgeAt)
    {
        string token = Token(row, judgeAt);
        return token.Equals("NG", StringComparison.OrdinalIgnoreCase) || token == "不合格";
    }

    static bool IsOk(string row, int? judgeAt)
    {
        string token = Token(row, judgeAt);
        return token.Equals("OK", StringComparison.OrdinalIgnoreCase) || token == "合格";
    }

    static string OpenName(string row, int? fileAt, int? judgeAt)
    {
        if (fileAt == null || !IsNg(row, judgeAt))
            return "";
        string[] cells = Cells(row);
        if (fileAt.Value >= cells.Length)
            return "";
        try
        {
            return NgFilename(cells[fileAt.Value]);
        }
        catch (InvalidOperationException)
        {
            return "";
        }
    }

    static string NgFilename(string name)
    {
        string filename = name.Trim().Trim('\'', '"');
        if (filename.Length == 0 || filename is "." or ".." || filename.Contains('/') || filename.Contains('\\') || Path.GetFileName(filename) != filename)
            throw new InvalidOperationException("不合格圖的檔名不正確");
        if (!Suffixes.Contains(Path.GetExtension(filename)))
            throw new InvalidOperationException("不合格圖必須是影像檔");
        return filename;
    }

    static string Yield(int ok, int total)
    {
        if (total <= 0)
            return "";
        double rounded = Math.Round(1000.0 * ok / total, MidpointRounding.ToEven) / 10;
        if (Math.Abs(rounded - Math.Round(rounded)) < 1e-9)
            return Math.Round(rounded).ToString(CultureInfo.InvariantCulture);
        return rounded.ToString("0.#", CultureInfo.InvariantCulture);
    }
}
