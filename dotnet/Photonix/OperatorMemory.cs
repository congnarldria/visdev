using System.IO;
using System.Text.Json;

namespace Photonix;

/// <summary>記住最近用過的算子，以及每個參數上次填的值。</summary>
sealed class OperatorMemory
{
    public Dictionary<string, Dictionary<string, string>> Fields { get; set; } = new();
    public List<string> Recent { get; set; } = [];

    public static OperatorMemory Load()
    {
        try
        {
            string path = FilePath();
            if (!File.Exists(path))
                return new OperatorMemory();
            return JsonSerializer.Deserialize<OperatorMemory>(File.ReadAllText(path)) ?? new OperatorMemory();
        }
        catch (Exception)
        {
            return new OperatorMemory();
        }
    }

    public void Remember(OperatorDoc op, IReadOnlyList<string> values)
    {
        var fields = new Dictionary<string, string>();
        for (int i = 0; i < op.Params.Count && i < values.Count; i++)
        {
            string value = values[i].Trim();
            if (value.Length > 0)
                fields[op.Params[i].Name] = value;
        }
        Fields[op.Name] = fields;
        Recent.Remove(op.Name);
        Recent.Insert(0, op.Name);
        if (Recent.Count > 8)
            Recent.RemoveRange(8, Recent.Count - 8);
    }

    public void Save()
    {
        try
        {
            string path = FilePath();
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, JsonSerializer.Serialize(this));
        }
        catch (Exception)
        {
            // 記不住時，這次對話裡的選單仍會更新。
        }
    }

    static string FilePath() => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Photonix",
        "operator-memory.json");
}
