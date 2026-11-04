using System.Text;
using System.Text.RegularExpressions;
using System.Xml.Linq;

namespace Photonix;

/// <summary>把 HDevelop 的 .hdev / .hdvp / .hdpl 轉成 Photonix 程式。有密碼的程序不打開。</summary>
static class HDevelopFile
{
    const string Locked = "* 這份 HALCON 程式是二進位，或整份被密碼鎖住，內容沒有打開。\n";

    static readonly HashSet<string> Skip = new(StringComparer.OrdinalIgnoreCase)
    {
        "if", "else", "endif", "for", "endfor", "while", "endwhile",
        "try", "catch", "endtry", "procedure", "endprocedure", "endproc", "return",
    };

    static readonly Dictionary<string, string> Groups = new(StringComparer.OrdinalIgnoreCase)
    {
        ["io"] = "iconic_in",
        ["ii"] = "iconic_in",
        ["oo"] = "iconic_out",
        ["ic"] = "control_in",
        ["ci"] = "control_in",
        ["oc"] = "control_out",
        ["co"] = "control_out",
    };

    public static (string Source, List<string> Skipped) Convert(byte[] raw, IReadOnlyDictionary<string, string> operators)
    {
        string? text = Decode(raw);
        if (text == null)
            return (Locked, ["整份程式"]);
        string stripped = text.TrimStart('\uFEFF', ' ', '\t', '\r', '\n');
        if (stripped.StartsWith("<?xml", StringComparison.Ordinal) || stripped.StartsWith("<hdevelop", StringComparison.OrdinalIgnoreCase))
            return FromXml(text, operators);
        return (FromLines(text.Replace("\r\n", "\n").Replace('\r', '\n').Split('\n'), [], new HashSet<string>(StringComparer.OrdinalIgnoreCase), operators), []);
    }

    static string? Decode(byte[] raw)
    {
        if (raw.Length == 0)
            return "";
        if (raw.Length >= 2 && raw[0] == 0xFF && raw[1] == 0xFE)
            return Encoding.Unicode.GetString(raw);
        if (raw.Length >= 2 && raw[0] == 0xFE && raw[1] == 0xFF)
            return Encoding.BigEndianUnicode.GetString(raw);
        if (raw.Length >= 3 && raw[0] == 0xEF && raw[1] == 0xBB && raw[2] == 0xBF)
            raw = raw[3..];
        int probe = Math.Min(400, raw.Length);
        for (int i = 0; i < probe; i++)
        {
            if (raw[i] == 0)
                return null;
        }
        try
        {
            return new UTF8Encoding(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true).GetString(raw);
        }
        catch (Exception)
        {
            return null;
        }
    }

    static (string Source, List<string> Skipped) FromXml(string text, IReadOnlyDictionary<string, string> operators)
    {
        XDocument document;
        try
        {
            document = XDocument.Parse(text);
        }
        catch (System.Xml.XmlException)
        {
            return (Locked, ["整份程式"]);
        }
        var procedures = document.Descendants().Where(node => Local(node) == "procedure").ToList();
        if (procedures.Count == 0)
            return (Locked, ["整份程式"]);
        var skipped = new List<string>();
        var plain = new List<XElement>();
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (XElement proc in procedures)
        {
            string name = (string?)proc.Attribute("name") ?? "";
            name = name.Trim();
            if (name.Length == 0)
                name = "procedure";
            if (IsProtected(proc))
            {
                skipped.Add(name);
                continue;
            }
            names.Add(name);
            plain.Add(proc);
        }
        string source = FromLines([], plain, names, operators);
        if (skipped.Count > 0)
        {
            string notes = string.Join("\n", skipped.Select(name => "* 程序 " + name + " 有密碼，沒有打開"));
            source = notes + (source.Trim().Length > 0 ? "\n" + source : "\n");
        }
        return (source, skipped);
    }

    static string FromLines(IReadOnlyList<string> lines, List<XElement> procedures, HashSet<string> procNames, IReadOnlyDictionary<string, string> operators)
    {
        var main = new List<string>();
        var others = new List<string>();
        if (procedures.Count > 0)
        {
            foreach (XElement proc in procedures)
            {
                string name = ((string?)proc.Attribute("name") ?? "").Trim();
                List<string> body = Rewrite(BodyLines(proc), operators, procNames);
                if (name.Equals("main", StringComparison.OrdinalIgnoreCase))
                    main.AddRange(body);
                else
                {
                    others.Add(Interface(proc));
                    others.AddRange(body);
                    others.Add("endprocedure");
                    others.Add("");
                }
            }
        }
        else
            main.AddRange(Rewrite(lines, operators, procNames));
        if (main.Count > 0 && others.Count > 0)
            main.Add("");
        main.AddRange(others);
        string text = string.Join("\n", main).Trim();
        return text.Length == 0 ? "\n" : text + "\n";
    }

    static List<string> Rewrite(IReadOnlyList<string> lines, IReadOnlyDictionary<string, string> operators, HashSet<string> procNames)
    {
        var written = new List<string>();
        foreach (string line in lines)
            written.AddRange(PrefixLine(line, operators, procNames));
        return written;
    }

    static List<string> PrefixLine(string line, IReadOnlyDictionary<string, string> operators, HashSet<string> procNames)
    {
        (string code, string comment) = SlashComment(line);
        var output = new List<string>();
        if (comment.Length > 0)
            output.Add("* " + comment);
        Match matched = Regex.Match(code, @"^(\s*)([A-Za-z_][\w.]*)(\s*\(.*)\s*$", RegexOptions.Singleline);
        if (matched.Success)
        {
            string name = matched.Groups[2].Value;
            string low = name.ToLowerInvariant();
            if (!Skip.Contains(low) && !procNames.Contains(low) && !low.StartsWith("vizion.", StringComparison.Ordinal))
            {
                if (operators.TryGetValue(low, out string? publicName))
                    code = matched.Groups[1].Value + publicName + matched.Groups[3].Value;
            }
        }
        if (code.Trim().Length > 0 || comment.Length == 0)
            output.Add(code);
        return output;
    }

    static (string Code, string Comment) SlashComment(string text)
    {
        char quote = '\0';
        for (int index = 0; index < text.Length - 1; index++)
        {
            char ch = text[index];
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
            if (text[index] == '/' && text[index + 1] == '/')
                return (text[..index].TrimEnd(), text[(index + 2)..].Trim());
        }
        return (text, "");
    }

    static string Interface(XElement proc)
    {
        var slots = new Dictionary<string, List<string>>
        {
            ["iconic_in"] = [],
            ["iconic_out"] = [],
            ["control_in"] = [],
            ["control_out"] = [],
        };
        XElement? face = proc.Elements().FirstOrDefault(node => Local(node) == "interface");
        if (face != null)
        {
            foreach (XElement group in face.Elements())
            {
                if (!Groups.TryGetValue(Local(group), out string? key))
                    continue;
                slots[key].AddRange(ParNames(group));
            }
        }
        string name = ((string?)proc.Attribute("name") ?? "").Trim();
        if (name.Length == 0)
            name = "procedure";
        return "procedure " + name + " ("
            + string.Join(", ", slots["iconic_in"]) + " : "
            + string.Join(", ", slots["iconic_out"]) + " : "
            + string.Join(", ", slots["control_in"]) + " : "
            + string.Join(", ", slots["control_out"]) + ")";
    }

    static List<string> ParNames(XElement group)
    {
        var names = new List<string>();
        string? own = (string?)group.Attribute("name");
        if (!string.IsNullOrWhiteSpace(own))
            names.Add(own.Trim());
        foreach (XElement node in group.Descendants())
        {
            if (Local(node) != "par")
                continue;
            string name = ((string?)node.Attribute("name") ?? "").Trim();
            if (name.Length > 0)
                names.Add(name);
        }
        return names;
    }

    static List<string> BodyLines(XElement proc)
    {
        var lines = new List<string>();
        XElement? body = proc.Elements().FirstOrDefault(node => Local(node) == "body");
        if (body == null)
            return lines;
        foreach (XElement node in body.Elements())
        {
            string tag = Local(node);
            string text = string.Concat(node.DescendantNodes().OfType<XText>().Select(item => item.Value));
            text = text.Replace("\r\n", "\n").Replace('\r', '\n').Trim('\n').TrimEnd();
            if (tag == "c")
            {
                string content = text.Trim();
                if (content.Length == 0)
                    lines.Add("");
                else if (content.StartsWith('*'))
                    lines.Add(content);
                else if (content.StartsWith("//"))
                    lines.Add("* " + content[2..].Trim());
                else
                    lines.Add("* " + content);
            }
            else if (tag == "l")
                lines.Add(text);
        }
        return lines;
    }

    static bool IsProtected(XElement proc)
    {
        string blob = string.Join(" ", proc.Attributes().Select(attr => attr.Name.LocalName + " " + attr.Value)).ToLowerInvariant();
        if (blob.Contains("password") || blob.Contains("protected") || blob.Contains("crypted") || blob.Contains("encrypted") || blob.Contains("locked"))
            return true;
        foreach (XElement node in proc.DescendantsAndSelf())
        {
            string tag = Local(node);
            if (tag == "xb" || tag is "protection" or "crypted" or "encrypted" or "password")
                return true;
            string kind = ((string?)node.Attribute("type") ?? "").ToLowerInvariant();
            if (kind is "protected" or "crypted" or "encrypted" or "password")
                return true;
            string text = (node.Value ?? "").ToLowerInvariant();
            if (text.Contains("password protected") || text.Contains("password-protected"))
                return true;
        }
        return false;
    }

    static string Local(XElement node) => node.Name.LocalName.ToLowerInvariant();
}
