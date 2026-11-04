using System.Text;

namespace Photonix;

static class ProgramFile
{
    public static readonly string[] Suffixes = [".viz", ".hdv", ".hdev", ".hdvp", ".hdpl", ".dev", ".dvp", ".txt"];

    public static bool IsHalcon(string name)
    {
        string lower = name.ToLowerInvariant();
        return lower.EndsWith(".hdev") || lower.EndsWith(".hdvp") || lower.EndsWith(".hdpl")
            || lower.EndsWith(".dev") || lower.EndsWith(".dvp");
    }

    public static string EnsureName(string? name)
    {
        string trimmed = string.IsNullOrWhiteSpace(name) ? "program.viz" : name.Trim();
        string lower = trimmed.ToLowerInvariant();
        if (lower.EndsWith(".hdv") || lower.EndsWith(".viz"))
            return trimmed;
        return trimmed + ".viz";
    }

    public static string Normalize(string? text) => (text ?? "").Replace("\r\n", "\n").Replace('\r', '\n');

    public static string ForDisk(string editorText, string path)
    {
        string text = Normalize(editorText);
        if (text.Length == 0 || !text.EndsWith('\n'))
            text += "\n";
        if (path.EndsWith(".viz", StringComparison.OrdinalIgnoreCase))
            text = text.Replace("\n", "\r\n");
        return text;
    }
}
