using System.Text.RegularExpressions;

namespace Photonix;

/// <summary>和網頁版 operator_call.js 同一套：拆參數、組呼叫、改一行。</summary>
static class OperatorCall
{
    public static string[] SplitCall(string text)
    {
        var parts = new List<string>();
        var buf = new System.Text.StringBuilder();
        char quote = '\0';
        int depth = 0;
        foreach (char ch in text ?? "")
        {
            if (quote != '\0')
            {
                buf.Append(ch);
                if (ch == quote)
                    quote = '\0';
                continue;
            }
            if (ch is '\'' or '"')
            {
                quote = ch;
                buf.Append(ch);
                continue;
            }
            if (ch is '[' or '(')
            {
                depth++;
                buf.Append(ch);
                continue;
            }
            if (ch is ']' or ')')
            {
                depth--;
                buf.Append(ch);
                continue;
            }
            if (ch == ',' && depth == 0)
            {
                parts.Add(buf.ToString().Trim());
                buf.Clear();
                continue;
            }
            buf.Append(ch);
        }
        if (buf.ToString().Trim().Length > 0)
            parts.Add(buf.ToString().Trim());
        return parts.ToArray();
    }

    public static string[] ExampleTokens(OperatorDoc op)
    {
        string example = op.Example ?? "";
        int open = example.IndexOf('(');
        int close = example.LastIndexOf(')');
        if (open < 0 || close < open)
            return [];
        return SplitCall(example[(open + 1)..close]);
    }

    public static string[] CallValues(OperatorDoc op, string[]? tokens)
    {
        string[] source = tokens ?? ExampleTokens(op);
        var values = new string[op.Params.Count];
        for (int i = 0; i < op.Params.Count; i++)
        {
            string value = i < source.Length ? source[i] : "";
            values[i] = string.IsNullOrEmpty(value) ? Fallback(op.Params[i]) : value;
        }
        return values;
    }

    public static string[] DialogValues(OperatorDoc op, string[]? preset, bool editing, IReadOnlyDictionary<string, string>? memory)
    {
        if (preset != null)
            return CallValues(op, preset);
        if (editing)
            return CallValues(op, null);
        string[] values = CallValues(op, null);
        if (memory == null)
            return values;
        for (int i = 0; i < op.Params.Count; i++)
        {
            if (memory.TryGetValue(op.Params[i].Name, out string? saved) && !string.IsNullOrEmpty(saved))
                values[i] = saved;
        }
        return values;
    }

    public static string FormatCall(string name, IReadOnlyList<string> values) =>
        name + " (" + string.Join(", ", values) + ")";

    public static ParsedCall? ParseLine(string text)
    {
        string src = (text ?? "").Trim();
        if (src.Length == 0 || src.StartsWith('*'))
            return null;
        Match match = Regex.Match(src, @"^((?:Vizion\.)?[A-Za-z_][A-Za-z0-9_]*)\s*\(", RegexOptions.IgnoreCase);
        if (!match.Success)
            return null;
        char quote = '\0';
        int depth = 1;
        var body = new System.Text.StringBuilder();
        for (int i = match.Length; i < src.Length; i++)
        {
            char ch = src[i];
            if (quote != '\0')
            {
                body.Append(ch);
                if (ch == quote)
                    quote = '\0';
                continue;
            }
            if (ch is '\'' or '"')
            {
                quote = ch;
                body.Append(ch);
                continue;
            }
            if (ch == '(')
            {
                depth++;
                body.Append(ch);
                continue;
            }
            if (ch == ')')
            {
                depth--;
                if (depth == 0)
                    return new ParsedCall(match.Groups[1].Value, SplitCall(body.ToString()));
                body.Append(ch);
                continue;
            }
            body.Append(ch);
        }
        return null;
    }

    public static (string Text, int Caret) ReplaceLine(string source, int lineIndex, string call)
    {
        string[] lines = (source ?? "").Replace("\r\n", "\n").Replace('\r', '\n').Split('\n');
        if (lineIndex < 0 || lineIndex >= lines.Length)
            return (source ?? "", 0);
        string current = lines[lineIndex];
        int indent = 0;
        while (indent < current.Length && char.IsWhiteSpace(current[indent]))
            indent++;
        lines[lineIndex] = current[..indent] + call;
        string text = string.Join("\n", lines);
        int caret = string.Join("\n", lines.Take(lineIndex + 1)).Length;
        return (text, caret);
    }

    static string Fallback(OperatorParam param)
    {
        if (param.Role == "控制輸入")
            return param.Default ?? "";
        return param.Name ?? "";
    }
}

readonly record struct ParsedCall(string Name, string[] Args);
