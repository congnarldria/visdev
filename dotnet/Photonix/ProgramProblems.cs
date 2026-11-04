using System.Text.RegularExpressions;

namespace Photonix;

sealed class Problem
{
    public string Sev { get; init; } = "";
    public string Severity { get; init; } = "";
    public string Code { get; init; } = "";
    public string Text { get; init; } = "";
    public string File { get; init; } = "";
    public int Line { get; init; }
    public string LineText => Line > 0 ? Line.ToString() : "";
}

static class ProgramProblems
{
    static readonly HashSet<string> Keywords = new(StringComparer.OrdinalIgnoreCase)
    {
        "if", "else", "endif", "for", "endfor", "while", "endwhile",
        "try", "catch", "endtry", "procedure", "endprocedure", "endproc",
    };

    public static List<Problem> Collect(
        string source,
        string file,
        IReadOnlyList<OperatorDoc> catalog,
        string? runError,
        int errorLine,
        IReadOnlyList<string> notes)
    {
        var items = new List<Problem>();
        string[] lines = (source ?? "").Replace("\r\n", "\n").Replace('\r', '\n').Split('\n');
        var procedures = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (string raw in lines)
        {
            Match header = Regex.Match(raw.Trim(), @"^procedure\s+([A-Za-z_]\w*)", RegexOptions.IgnoreCase);
            if (header.Success)
                procedures.Add(header.Groups[1].Value);
        }
        var reads = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var writes = new List<(string Name, int Line)>();
        for (int index = 0; index < lines.Length; index++)
        {
            int lineNo = index + 1;
            string text = lines[index].Trim();
            if (text.Length == 0 || text.StartsWith('*') || text.StartsWith("//"))
                continue;
            Match loop = Regex.Match(text, @"^for\s+([A-Za-z_]\w*)\s*:=(.*)$", RegexOptions.IgnoreCase);
            if (loop.Success)
            {
                writes.Add((loop.Groups[1].Value, lineNo));
                NoteReads(loop.Groups[2].Value, reads);
                continue;
            }
            Match dotted = Regex.Match(text, @"^(Vizion\.[A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", RegexOptions.IgnoreCase);
            Match bare = Regex.Match(text, @"^([A-Za-z_]\w*)\s*\((.*)\)\s*$");
            if (dotted.Success)
            {
                OperatorDoc? op = Find(catalog, dotted.Groups[1].Value);
                if (op == null)
                    items.Add(Error("VIZ002", "未知算子 " + dotted.Groups[1].Value, file, lineNo));
                else
                    NoteCall(op, dotted.Groups[2].Value, lineNo, reads, writes);
                continue;
            }
            if (bare.Success)
            {
                string low = bare.Groups[1].Value.ToLowerInvariant();
                if (Keywords.Contains(low) || procedures.Contains(bare.Groups[1].Value))
                {
                    NoteReads(bare.Groups[2].Value, reads);
                    continue;
                }
                OperatorDoc? op = Find(catalog, bare.Groups[1].Value);
                if (op != null)
                {
                    items.Add(Error("VIZ001", "算子要寫成 " + op.Name + " (...)", file, lineNo));
                    NoteCall(op, bare.Groups[2].Value, lineNo, reads, writes);
                }
                else
                    items.Add(Error("VIZ002", "未知算子 " + bare.Groups[1].Value, file, lineNo));
                continue;
            }
            Match assign = Regex.Match(text, @"^([A-Za-z_]\w*)\s*:=(.*)$");
            if (assign.Success)
            {
                writes.Add((assign.Groups[1].Value, lineNo));
                NoteReads(assign.Groups[2].Value, reads);
            }
        }
        var warned = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var item in writes)
        {
            if (!warned.Add(item.Name) || reads.Contains(item.Name))
                continue;
            items.Add(new Problem
            {
                Sev = "warning",
                Severity = "警告",
                Code = "VIZ100",
                Text = "變數 " + item.Name + " 指定了但沒有被使用",
                File = file,
                Line = item.Line,
            });
        }
        foreach (string name in procedures)
        {
            bool called = lines.Any(raw => Regex.IsMatch(raw.Trim(), "^" + Regex.Escape(name) + @"\s*\(", RegexOptions.IgnoreCase));
            if (called)
                continue;
            int line = 1;
            for (int i = 0; i < lines.Length; i++)
            {
                if (Regex.IsMatch(lines[i].Trim(), @"^procedure\s+" + Regex.Escape(name) + @"\b", RegexOptions.IgnoreCase))
                {
                    line = i + 1;
                    break;
                }
            }
            items.Add(new Problem
            {
                Sev = "warning",
                Severity = "警告",
                Code = "VIZ101",
                Text = "程序 " + name + " 沒有被呼叫",
                File = file,
                Line = line,
            });
        }
        foreach (string note in notes)
        {
            string text = note.Trim();
            Match where = Regex.Match(text, @"第\s*(\d+)\s*行");
            int lineNo = where.Success ? int.Parse(where.Groups[1].Value) : 0;
            if (text.StartsWith("錯誤"))
            {
                int line = lineNo > 0 ? lineNo : errorLine;
                if (items.Any(item => item.Sev == "error" && item.Line == line))
                    continue;
                items.Add(Error("VIZ200", text["錯誤".Length..].Trim(), file, line));
            }
            else if (text.StartsWith("catch "))
                items.Add(Message("VIZ300", text, file, lineNo));
            else if (text.Contains("執行完成") || text.StartsWith("中斷點") || text.StartsWith("已停止") || text.Contains("Vizion.stop"))
                items.Add(Message("VIZ301", text, file, lineNo));
        }
        if (!string.IsNullOrEmpty(runError) && !items.Any(item => item.Sev == "error" && item.Line == errorLine && item.Text.Contains(runError)))
            items.Add(Error("VIZ200", runError, file, errorLine));
        return items;
    }

    static Problem Error(string code, string text, string file, int line) => new()
    {
        Sev = "error",
        Severity = "錯誤",
        Code = code,
        Text = text,
        File = file,
        Line = line,
    };

    static Problem Message(string code, string text, string file, int line) => new()
    {
        Sev = "message",
        Severity = "訊息",
        Code = code,
        Text = text,
        File = file,
        Line = line,
    };

    static OperatorDoc? Find(IReadOnlyList<OperatorDoc> catalog, string token)
    {
        string lower = token.ToLowerInvariant();
        return catalog.FirstOrDefault(op =>
        {
            string name = op.Name.ToLowerInvariant();
            return name == lower || name == "vizion." + lower;
        });
    }

    static void NoteReads(string expr, HashSet<string> reads)
    {
        var skip = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "and", "or", "not", "mod", "true", "false" };
        foreach (Match match in Regex.Matches(expr, @"[A-Za-z_]\w*"))
        {
            if (!skip.Contains(match.Value))
                reads.Add(match.Value);
        }
    }

    static void NoteCall(OperatorDoc op, string argsBody, int lineNo, HashSet<string> reads, List<(string Name, int Line)> writes)
    {
        string[] args = OperatorCall.SplitCall(argsBody);
        for (int i = 0; i < op.Params.Count; i++)
        {
            if (i >= args.Length || args[i].Length == 0)
                continue;
            bool ident = Regex.IsMatch(args[i], @"^[A-Za-z_]\w*$");
            if (op.Params[i].Role.Contains("輸出"))
            {
                if (ident)
                    writes.Add((args[i], lineNo));
            }
            else
                NoteReads(args[i], reads);
        }
    }
}
