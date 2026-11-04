using System.IO;
using System.Reflection;
using System.Text.Json;

namespace Photonix;

/// <summary>和網頁版 /api/operators 同一份目錄，由 list_operators() 匯出。</summary>
static class OperatorCatalog
{
    public static OperatorDoc[] Load()
    {
        var assembly = typeof(OperatorCatalog).Assembly;
        using Stream stream = assembly.GetManifestResourceStream("Photonix.operators.json")
            ?? throw new InvalidOperationException("找不到算子目錄");
        return JsonSerializer.Deserialize<OperatorDoc[]>(stream, new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
            ?? [];
    }
}

sealed class OperatorDoc
{
    public string Name { get; set; } = "";
    public string Group { get; set; } = "";
    public string Short { get; set; } = "";
    public string Purpose { get; set; } = "";
    public string Principle { get; set; } = "";
    public string Setup { get; set; } = "";
    public string Prototype { get; set; } = "";
    public string Example { get; set; } = "";
    public string Note { get; set; } = "";
    public List<OperatorParam> Params { get; set; } = [];
}

sealed class OperatorParam
{
    public string Name { get; set; } = "";
    public string Role { get; set; } = "";
    public string Desc { get; set; } = "";
    public string? Default { get; set; }
    public List<string> Choices { get; set; } = [];
    public string Browse { get; set; } = "";
}
