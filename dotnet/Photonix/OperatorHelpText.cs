using System.Text;

namespace Photonix;

/// <summary>目錄、參數視窗和程式編輯器共用的說明。文字只來自算子目錄。</summary>
static class OperatorHelpText
{
    public static string Format(OperatorDoc op)
    {
        var text = new StringBuilder();
        text.Append("用途\n");
        text.Append(op.Purpose ?? "");
        text.Append("\n\n原理\n");
        text.Append(op.Principle ?? "");
        text.Append("\n\n用法\n");
        text.Append(op.Example ?? "");
        if (op.Params.Count > 0)
        {
            text.Append("\n\n參數");
            foreach (OperatorParam param in op.Params)
            {
                text.Append('\n');
                text.Append(param.Name);
                text.Append('：');
                text.Append(param.Desc ?? "");
            }
        }
        return text.ToString();
    }
}
