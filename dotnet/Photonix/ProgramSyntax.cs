using System.Text.RegularExpressions;

namespace Photonix;

enum ProgramTokenKind { Plain, Comment, String, Number, Key, Operator }

readonly record struct ProgramToken(string Text, ProgramTokenKind Kind);

readonly record struct SenseItem(string Kind, string Label, string Detail, string Insert, int? Caret);

readonly record struct SenseChunk(string Token, int TokenStart, string Head, string? ParenBefore, string? ParenInside, bool ParenFollows);

static class ProgramSyntax
{
    static readonly Regex TokenPattern = new(
        @"('[^']*'|""[^""]*""|[A-Za-z_]\w*\.[A-Za-z_]\w*|\b(?:if|else|endif|for|endfor|while|endwhile)\b|\b\d+(?:\.\d+)?\b|[A-Za-z_]\w*|\s+|.)",
        RegexOptions.Compiled);

    static readonly (string Name, string Insert, int? Caret, string Detail)[] Keys =
    [
        ("if", "if ()", 4, "條件成立才執行，直到 endif"),
        ("else", "else", null, "條件不成立時的分支"),
        ("endif", "endif", null, "結束 if"),
        ("for", "for Index := 1 to 10 by 1", null, "從起點數到終點，步進 by"),
        ("endfor", "endfor", null, "結束 for"),
        ("while", "while ()", 7, "條件成立就重複，直到 endwhile"),
        ("endwhile", "endwhile", null, "結束 while"),
    ];

    public static IReadOnlyList<ProgramToken> Tokens(string line, HashSet<string> operators)
    {
        if (Regex.IsMatch(line, @"^\s*(?:\*|//)"))
            return [new ProgramToken(line, ProgramTokenKind.Comment)];
        var tokens = new List<ProgramToken>();
        foreach (Match match in TokenPattern.Matches(line))
        {
            string token = match.Value;
            ProgramTokenKind kind = ProgramTokenKind.Plain;
            if (token.Length > 0 && token[0] is '\'' or '"')
                kind = ProgramTokenKind.String;
            else if (token.Length > 0 && char.IsDigit(token[0]))
                kind = ProgramTokenKind.Number;
            else if (token.Equals("if", StringComparison.Ordinal) || token.Equals("else", StringComparison.Ordinal)
                || token.Equals("endif", StringComparison.Ordinal) || token.Equals("for", StringComparison.Ordinal)
                || token.Equals("endfor", StringComparison.Ordinal) || token.Equals("while", StringComparison.Ordinal)
                || token.Equals("endwhile", StringComparison.Ordinal))
                kind = ProgramTokenKind.Key;
            else if (operators.Contains(token.ToLowerInvariant()))
                kind = ProgramTokenKind.Operator;
            tokens.Add(new ProgramToken(token, kind));
        }
        return tokens;
    }

    public static SenseChunk? Chunk(string value, int pos, int selectionEnd)
    {
        if (pos != selectionEnd)
            return null;
        int lineStart = pos == 0 ? 0 : value.LastIndexOf('\n', pos - 1) + 1;
        while (lineStart > 0)
        {
            int prevEnd = lineStart - 1;
            int prevStart = prevEnd == 0 ? 0 : value.LastIndexOf('\n', prevEnd - 1) + 1;
            if (!value[prevStart..prevEnd].TrimEnd().EndsWith('\\'))
                break;
            lineStart = prevStart;
        }
        string chunk = value[lineStart..pos];
        string trimmed = chunk.TrimStart();
        if (trimmed.StartsWith('*') || trimmed.StartsWith("//"))
            return null;
        if (QuoteOpen(chunk))
            return null;
        Match tokenMatch = Regex.Match(chunk, @"(?:Vizion\.)?[A-Za-z_][A-Za-z0-9_]*$", RegexOptions.IgnoreCase);
        string token = tokenMatch.Success ? tokenMatch.Value : "";
        string head = chunk[..(chunk.Length - token.Length)];
        var paren = UnclosedParen(head);
        bool parenFollows = Regex.IsMatch(value[pos..], @"^\s*\(");
        return new SenseChunk(token, pos - token.Length, head, paren?.Before, paren?.Inside, parenFollows);
    }

    public static (string Signature, IReadOnlyList<SenseItem> Items) Suggest(
        SenseChunk chunk, IReadOnlyList<OperatorDoc> catalog, string program, bool force)
    {
        string lower = chunk.Token.ToLowerInvariant();
        if (chunk.ParenBefore != null)
        {
            Match nameMatch = Regex.Match(chunk.ParenBefore, @"(?:^|\s)((?:Vizion\.)?[A-Za-z_][A-Za-z0-9_]*)\s*$", RegexOptions.IgnoreCase);
            string opName = nameMatch.Success ? nameMatch.Groups[1].Value : "";
            OperatorDoc? op = Find(catalog, opName);
            int argIndex = CommaCount(chunk.ParenInside ?? "");
            OperatorParam? param = op != null && argIndex < op.Params.Count ? op.Params[argIndex] : null;
            string exampleArg = "";
            if (op != null)
            {
                string[] args = OperatorCall.ExampleTokens(op);
                if (argIndex < args.Length)
                    exampleArg = args[argIndex];
            }
            (List<string> iconic, List<string> control) = Names(program, catalog);
            bool imageRole = param != null && param.Role.Contains("圖像");
            bool controlRole = param != null && param.Role.Contains("控制");
            var picked = new List<SenseItem>();
            var used = new HashSet<string>();
            void Add(string kind, string label, string detail)
            {
                if (label.Length == 0 || !used.Add(label))
                    return;
                if (lower.Length > 0 && !label.StartsWith(lower, StringComparison.OrdinalIgnoreCase))
                    return;
                picked.Add(new SenseItem(kind, label, detail, label, null));
            }
            if (exampleArg.Length > 0)
                Add("範本", exampleArg, param?.Desc ?? "這個參數的寫法");
            IEnumerable<string> pool = imageRole ? iconic : controlRole ? control : iconic.Concat(control);
            var iconicSet = iconic.ToHashSet();
            foreach (string name in pool)
            {
                string detail = iconicSet.Contains(name) && !control.Contains(name) ? "圖像變數" : "控制變數";
                Add("變數", name, detail);
            }
            string signature = "";
            if (op != null)
            {
                signature = op.Name + " ( " + string.Join(", ", op.Params.Select(item => item.Name)) + " )";
                if (param != null)
                    signature += "　" + param.Desc;
            }
            return (signature, picked.Take(40).ToList());
        }
        string statement = Regex.Replace(chunk.Head, @"\\\s*\n", "").Trim();
        if (statement.Length > 0)
            return ("", []);
        if (chunk.Token.Length == 0 && !force)
            return ("", []);
        var ops = catalog
            .Where(op => MatchesPrefix(op, lower))
            .Where(op => !(chunk.ParenFollows && op.Name.Equals(lower, StringComparison.OrdinalIgnoreCase)))
            .OrderBy(op => op.Name.Length).ThenBy(op => op.Order)
            .Select(op => new SenseItem("算子", op.Name, op.Short, chunk.ParenFollows ? op.Name : op.Example, null));
        var keys = Keys
            .Where(item => item.Name.StartsWith(lower, StringComparison.Ordinal))
            .Where(item => !(chunk.ParenFollows && item.Name == lower))
            .OrderBy(item => item.Name.Length)
            .Select(item => new SenseItem("語法", item.Name, item.Detail, item.Insert, item.Caret));
        var items = ops.Concat(keys).OrderBy(item => item.Label.Length).Take(40).ToList();
        return ("", items);
    }

    public static bool MatchesPrefix(OperatorDoc op, string lower)
    {
        string name = op.Name.ToLowerInvariant();
        string bare = name.StartsWith("vizion.", StringComparison.Ordinal) ? name["vizion.".Length..] : name;
        return name.StartsWith(lower, StringComparison.Ordinal) || bare.StartsWith(lower, StringComparison.Ordinal);
    }

    static OperatorDoc? Find(IReadOnlyList<OperatorDoc> catalog, string token)
    {
        if (token.Length == 0)
            return null;
        string lower = token.ToLowerInvariant();
        return catalog.FirstOrDefault(op =>
        {
            string name = op.Name.ToLowerInvariant();
            return name == lower || name == "vizion." + lower;
        });
    }

    static (List<string> Iconic, List<string> Control) Names(string program, IReadOnlyList<OperatorDoc> catalog)
    {
        var iconic = new List<string>();
        var control = new List<string>();
        var seenI = new HashSet<string>();
        var seenC = new HashSet<string>();
        foreach (string raw in LogicalLines(program))
        {
            if (raw.Length == 0 || raw.StartsWith('*') || raw.StartsWith("//"))
                continue;
            Match assign = Regex.Match(raw, @"^([A-Za-z_]\w*)\s*:=");
            if (assign.Success)
            {
                if (seenC.Add(assign.Groups[1].Value))
                    control.Add(assign.Groups[1].Value);
                continue;
            }
            Match call = Regex.Match(raw, @"^((?:Vizion\.)?[A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", RegexOptions.IgnoreCase);
            if (!call.Success)
                continue;
            OperatorDoc? op = Find(catalog, call.Groups[1].Value);
            if (op == null)
                continue;
            string[] args = OperatorCall.SplitCall(call.Groups[2].Value);
            for (int i = 0; i < args.Length && i < op.Params.Count; i++)
            {
                if (!Regex.IsMatch(args[i], @"^[A-Za-z_]\w*$"))
                    continue;
                if (op.Params[i].Role.Contains("圖像"))
                {
                    if (seenI.Add(args[i]))
                        iconic.Add(args[i]);
                }
                else if (seenC.Add(args[i]))
                    control.Add(args[i]);
            }
        }
        return (iconic, control);
    }

    static IEnumerable<string> LogicalLines(string source)
    {
        string pending = "";
        foreach (string line in source.Replace("\r\n", "\n").Split('\n'))
        {
            string trimmed = line.Trim();
            if (trimmed.EndsWith('\\'))
            {
                pending += trimmed[..^1] + " ";
                continue;
            }
            yield return (pending + line).Trim();
            pending = "";
        }
        if (pending.Length > 0)
            yield return pending.Trim();
    }

    static bool QuoteOpen(string text)
    {
        char quote = '\0';
        foreach (char ch in text)
        {
            if (quote != '\0')
            {
                if (ch == quote)
                    quote = '\0';
            }
            else if (ch is '\'' or '"')
                quote = ch;
        }
        return quote != '\0';
    }

    static (string Before, string Inside)? UnclosedParen(string text)
    {
        char quote = '\0';
        int depth = 0;
        int openAt = -1;
        for (int i = 0; i < text.Length; i++)
        {
            char ch = text[i];
            if (quote != '\0')
            {
                if (ch == quote)
                    quote = '\0';
                continue;
            }
            if (ch is '\'' or '"')
            {
                quote = ch;
                continue;
            }
            if (ch is '(' or '[')
            {
                if (depth == 0 && ch == '(')
                    openAt = i;
                depth++;
            }
            else if (ch is ')' or ']')
            {
                depth--;
                if (depth <= 0)
                {
                    depth = 0;
                    openAt = -1;
                }
            }
        }
        if (openAt < 0)
            return null;
        return (text[..openAt], text[(openAt + 1)..]);
    }

    static int CommaCount(string text)
    {
        char quote = '\0';
        int depth = 0;
        int count = 0;
        foreach (char ch in text)
        {
            if (quote != '\0')
            {
                if (ch == quote)
                    quote = '\0';
                continue;
            }
            if (ch is '\'' or '"')
            {
                quote = ch;
                continue;
            }
            if (ch is '(' or '[')
                depth++;
            else if (ch is ')' or ']')
                depth = Math.Max(0, depth - 1);
            else if (ch == ',' && depth == 0)
                count++;
        }
        return count;
    }
}
