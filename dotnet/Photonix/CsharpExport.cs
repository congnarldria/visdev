using System.Globalization;
using System.Text.RegularExpressions;

namespace Photonix;

sealed class ExportException : Exception
{
    public ExportException(string message, int line = 0)
        : base(line > 0 ? "第 " + line + " 行：" + message : message)
    {
    }
}

/// <summary>和網頁版 export_cs.py 同一套：把讀圖、二值化、連通和區域算子匯出成 C#。</summary>
static class CsharpExport
{
    static readonly HashSet<string> Supported = new(StringComparer.Ordinal)
    {
        "read_image", "threshold", "connection", "count_obj", "area_center", "dev_display",
        "select_shape", "select_obj", "concat_obj", "union1", "union2", "intersection", "difference",
        "gen_rectangle1", "smallest_rectangle1", "get_image_size", "get_grayval", "test_region_point",
        "intensity", "dilation_rectangle1", "erosion_rectangle1", "opening_rectangle1", "closing_rectangle1",
        "dilation_circle", "erosion_circle", "opening_circle", "closing_circle", "boundary",
        "fill_up", "complement", "reduce_domain", "change_domain", "get_domain", "crop_rectangle1",
        "gen_image_const", "gen_image_proto", "paint_region", "overpaint_region", "set_grayval",
    };

    static readonly HashSet<string> Shape = new(StringComparer.Ordinal)
    {
        "area", "row", "column", "width", "height", "ratio", "center_row", "center_column", "center_col",
    };

    static readonly HashSet<string> Keywords = new(StringComparer.Ordinal)
    {
        "abstract", "as", "base", "bool", "break", "byte", "case", "catch", "char", "checked",
        "class", "const", "continue", "decimal", "default", "delegate", "do", "double", "else",
        "enum", "event", "explicit", "extern", "false", "finally", "fixed", "float", "for",
        "foreach", "goto", "if", "implicit", "in", "int", "interface", "internal", "is", "lock",
        "long", "namespace", "new", "null", "object", "operator", "out", "override", "params",
        "private", "protected", "public", "readonly", "ref", "return", "sbyte", "sealed",
        "short", "sizeof", "stackalloc", "static", "string", "struct", "switch", "this",
        "throw", "true", "try", "typeof", "uint", "ulong", "unchecked", "unsafe", "ushort",
        "using", "var", "virtual", "void", "volatile", "while",
    };

    static readonly Regex IdentPattern = new("^[A-Za-z_]\\w*$", RegexOptions.CultureInvariant);
    static readonly Regex AssignPattern = new("^([A-Za-z_]\\w*)\\s*:=\\s*(.+)$", RegexOptions.Singleline | RegexOptions.CultureInvariant);
    static readonly Regex VisionCallPattern = new("^(Vizion\\.[A-Za-z_]\\w*)\\s*\\((.*)\\)\\s*$", RegexOptions.IgnoreCase | RegexOptions.Singleline | RegexOptions.CultureInvariant);
    static readonly Regex BareCallPattern = new("^([A-Za-z_]\\w*)\\s*\\((.*)\\)\\s*$", RegexOptions.Singleline | RegexOptions.CultureInvariant);
    static readonly Regex IfPattern = new("^if\\s*\\((.*)\\)\\s*$", RegexOptions.IgnoreCase | RegexOptions.Singleline | RegexOptions.CultureInvariant);
    static readonly Regex WhilePattern = new("^while\\s*\\((.*)\\)\\s*$", RegexOptions.IgnoreCase | RegexOptions.Singleline | RegexOptions.CultureInvariant);

    public static string ClassName(string? name)
    {
        string stem = (name ?? "").Trim();
        string lowered = stem.ToLowerInvariant();
        if (lowered.EndsWith(".viz", StringComparison.Ordinal) || lowered.EndsWith(".hdv", StringComparison.Ordinal))
        {
            int dot = stem.LastIndexOf('.');
            if (dot >= 0)
                stem = stem[..dot];
        }
        var cleaned = new System.Text.StringBuilder();
        foreach (char ch in stem)
        {
            if (ch is (>= 'A' and <= 'Z') or (>= 'a' and <= 'z') or (>= '0' and <= '9') or '_')
                cleaned.Append(ch);
        }
        string text = cleaned.ToString();
        if (text.Length == 0 || char.IsDigit(text[0]) || Keywords.Contains(text))
            text = "Exported" + text;
        return IdentPattern.IsMatch(text) ? text : "ExportedProgram";
    }

    public static string Export(string? source, string className, IEnumerable<OperatorDoc> operators)
    {
        if (!IdentPattern.IsMatch(className))
            throw new ExportException("類別名稱必須是 C# 識別字");
        var ops = new Dictionary<string, OperatorDoc>(StringComparer.OrdinalIgnoreCase);
        foreach (OperatorDoc op in operators)
        {
            int dot = op.Name.IndexOf('.');
            string bare = dot >= 0 ? op.Name[(dot + 1)..] : op.Name;
            ops[bare] = op;
        }
        string text = source ?? "";
        string[] rawLines = text.Replace("\r\n", "\n").Replace('\r', '\n').Split('\n');
        List<Stmt> program = Parse(rawLines, ops);
        return Emit(rawLines, program, className);
    }

    static string Emit(string[] rawLines, List<Stmt> program, string className)
    {
        var constants = new Dictionary<string, Value>(StringComparer.Ordinal);
        var images = new HashSet<string>(StringComparer.Ordinal);
        var regions = new Dictionary<string, string>(StringComparer.Ordinal);
        var kinds = new Dictionary<string, string>(StringComparer.Ordinal);
        var body = new List<string>();
        string lastImage = "";
        string lastRegion = "";
        string shownImage = "";
        string shownRegion = "";
        bool displayed = false;
        int measureIndex = 0;

        string Original(int line) => line > 0 && line <= rawLines.Length ? rawLines[line - 1].Trim() : "";

        void Assign(string name, string expr, string kind, int line)
        {
            if (!kinds.TryGetValue(name, out string? previous))
            {
                kinds[name] = kind;
                body.Add("        var " + Ident(name) + " = " + expr + ";");
                return;
            }
            if (previous != kind)
                throw new ExportException("變數 " + name + " 不能改成另一種結果", line);
            body.Add("        " + Ident(name) + " = " + expr + ";");
        }

        foreach (Stmt stmt in program)
        {
            if (stmt is AssignStmt assign)
            {
                Value value = Literal(assign.Expr, constants, assign.Line);
                if (value is ListValue)
                    throw new ExportException("匯出還沒有做 tuple 指定", assign.Line);
                if (value is not NumValue && value is not StrValue)
                    throw new ExportException("匯出只能指定數字或字串", assign.Line);
                constants[assign.Name] = value;
                continue;
            }
            if (stmt is not CallStmt call)
                throw new ExportException("匯出只做讀圖、二值化、連通和顯示", stmt.Line);
            if (!Supported.Contains(call.Bare))
                throw new ExportException("匯出還沒有做 " + call.Name, call.Line);
            string note = Original(call.Line);
            if (note.Length > 0)
                body.Add("        // " + note);
            string bare = call.Bare;
            if (bare == "read_image")
            {
                Value filename = Literal(call.ControlIn[0], constants, call.Line);
                if (filename is not StrValue text)
                    throw new ExportException("FileName 要是字串", call.Line);
                string name = call.IconicOut[0];
                Assign(name, "Operators.ReadImage(" + CsString(text.Text) + ")", "image", call.Line);
                images.Add(name);
                regions.Remove(name);
                lastImage = name;
                continue;
            }
            if (bare == "gen_image_const")
            {
                string kind = Shown(Literal(call.ControlIn[0], constants, call.Line)).Trim().Trim('\'', '"').ToLowerInvariant();
                if (kind != "byte")
                    throw new ExportException("C# 的 gen_image_const 目前只做 byte", call.Line);
                int wide = Whole(Literal(call.ControlIn[1], constants, call.Line), "Width", call.Line);
                int high = Whole(Literal(call.ControlIn[2], constants, call.Line), "Height", call.Line);
                string name = call.IconicOut[0];
                Assign(name, "Operators.Constant(" + wide + ", " + high + ")", "image", call.Line);
                images.Add(name);
                regions.Remove(name);
                lastImage = name;
                continue;
            }
            if (bare == "gen_image_proto")
            {
                string sourceName = call.IconicIn[0];
                if (!images.Contains(sourceName))
                    throw new ExportException("找不到影像 " + sourceName, call.Line);
                int gray = Whole(Literal(call.ControlIn[0], constants, call.Line), "Grayval", call.Line);
                string name = call.IconicOut[0];
                Assign(name, "Operators.Proto(" + Ident(sourceName) + ", " + gray + ")", "image", call.Line);
                images.Add(name);
                regions.Remove(name);
                lastImage = name;
                continue;
            }
            if (bare is "reduce_domain" or "change_domain")
            {
                string imageName = call.IconicIn[0];
                string regionName = call.IconicIn[1];
                if (!images.Contains(imageName))
                    throw new ExportException("找不到影像 " + imageName, call.Line);
                if (!regions.ContainsKey(regionName))
                    throw new ExportException("找不到區域 " + regionName, call.Line);
                string name = call.IconicOut[0];
                string method = bare == "reduce_domain" ? "ReduceDomain" : "ChangeDomain";
                Assign(name, "Operators." + method + "(" + Ident(imageName) + ", " + Ident(regionName) + ")", "image", call.Line);
                images.Add(name);
                regions.Remove(name);
                lastImage = name;
                continue;
            }
            if (bare == "get_domain")
            {
                string sourceName = call.IconicIn[0];
                if (!images.Contains(sourceName))
                    throw new ExportException("找不到影像 " + sourceName, call.Line);
                string name = call.IconicOut[0];
                Assign(name, "Operators.DomainOf(" + Ident(sourceName) + ")", "region", call.Line);
                regions[name] = "one";
                images.Remove(name);
                lastRegion = name;
                continue;
            }
            if (bare == "crop_rectangle1")
            {
                string sourceName = call.IconicIn[0];
                if (!images.Contains(sourceName))
                    throw new ExportException("找不到影像 " + sourceName, call.Line);
                int[] coords = Coords(call.ControlIn, constants, call.Line);
                string name = call.IconicOut[0];
                Assign(name, "Operators.Crop(" + Ident(sourceName) + ", " + string.Join(", ", coords) + ")", "image", call.Line);
                images.Add(name);
                regions.Remove(name);
                lastImage = name;
                continue;
            }
            if (bare is "paint_region" or "overpaint_region")
            {
                string regionName = call.IconicIn[0];
                string imageName = call.IconicIn[1];
                if (!regions.ContainsKey(regionName))
                    throw new ExportException("找不到區域 " + regionName, call.Line);
                if (!images.Contains(imageName))
                    throw new ExportException("找不到影像 " + imageName, call.Line);
                Value gray = Literal(call.ControlIn[0], constants, call.Line);
                string mode = Shown(Literal(call.ControlIn[1], constants, call.Line)).Trim().Trim('\'', '"').ToLowerInvariant();
                if (mode is not ("fill" or "margin"))
                    throw new ExportException("Type 只能是 fill 或 margin", call.Line);
                if (gray is ListValue)
                    throw new ExportException("匯出的塗色目前用一個灰階", call.Line);
                int value = Whole(gray, "Grayval", call.Line);
                string name = call.IconicOut[0];
                string method = bare == "paint_region" ? "Paint" : "Overpaint";
                Assign(name, "Operators." + method + "(" + Ident(regionName) + ", " + Ident(imageName) + ", " + value + ", " + CsString(mode) + ")", "image", call.Line);
                images.Add(name);
                regions.Remove(name);
                lastImage = name;
                continue;
            }
            if (bare == "set_grayval")
            {
                string sourceName = call.IconicIn[0];
                if (!images.Contains(sourceName))
                    throw new ExportException("找不到影像 " + sourceName, call.Line);
                Value row = Literal(call.ControlIn[0], constants, call.Line);
                Value column = Literal(call.ControlIn[1], constants, call.Line);
                Value gray = Literal(call.ControlIn[2], constants, call.Line);
                if (row is ListValue || column is ListValue || gray is ListValue)
                    throw new ExportException("匯出的 set_grayval 一次改一個灰階點", call.Line);
                string name = call.IconicOut[0];
                Assign(name, "Operators.SetGray(" + Ident(sourceName) + ", " + Rounded(row) + ", " + Rounded(column) + ", " + Whole(gray, "Grayval", call.Line) + ")", "image", call.Line);
                images.Add(name);
                regions.Remove(name);
                lastImage = name;
                continue;
            }
            if (bare == "threshold")
            {
                string sourceName = call.IconicIn[0];
                if (!images.Contains(sourceName))
                    throw new ExportException("找不到影像 " + sourceName, call.Line);
                int low = Whole(Literal(call.ControlIn[0], constants, call.Line), "MinGray", call.Line);
                int high = Whole(Literal(call.ControlIn[1], constants, call.Line), "MaxGray", call.Line);
                string name = call.IconicOut[0];
                Assign(name, "Operators.Threshold(" + Ident(sourceName) + ", " + low + ", " + high + ")", "region", call.Line);
                regions[name] = "one";
                images.Remove(name);
                lastRegion = name;
                continue;
            }
            if (bare == "connection")
            {
                string sourceName = call.IconicIn[0];
                if (!regions.TryGetValue(sourceName, out string? shape))
                    throw new ExportException("找不到區域 " + sourceName, call.Line);
                if (shape != "one")
                    throw new ExportException("連通的輸入要是還沒拆開的區域", call.Line);
                double distance = Scalar(Literal(call.ControlIn[0], constants, call.Line), "Distance", call.Line);
                if (distance < 0)
                    throw new ExportException("Distance 不能是負數", call.Line);
                string name = call.IconicOut[0];
                string expr = Math.Abs(distance - 1.0) < 1e-12
                    ? "Operators.Connection(" + Ident(sourceName) + ")"
                    : "Operators.Connection(" + Ident(sourceName) + ", " + CsNumber(distance) + ")";
                Assign(name, expr, "regions", call.Line);
                regions[name] = "many";
                images.Remove(name);
                lastRegion = name;
                continue;
            }
            if (bare is "union1" or "dilation_rectangle1" or "erosion_rectangle1" or "opening_rectangle1" or "closing_rectangle1" or "fill_up" or "complement" or "dilation_circle" or "erosion_circle" or "opening_circle" or "closing_circle" or "boundary")
            {
                string sourceName = call.IconicIn[0];
                if (!regions.TryGetValue(sourceName, out string? shape))
                    throw new ExportException("找不到區域 " + sourceName, call.Line);
                string name = call.IconicOut[0];
                string expr;
                if (bare == "union1")
                {
                    string source = shape == "many" ? Ident(sourceName) : "new[] { " + Ident(sourceName) + " }";
                    expr = "Operators.Union(" + source + ")";
                }
                else if (bare == "fill_up")
                    expr = "Operators.FillUp(" + Ident(sourceName) + ")";
                else if (bare == "complement")
                    expr = "Operators.Complement(" + Ident(sourceName) + ")";
                else if (bare is "dilation_circle" or "erosion_circle" or "opening_circle" or "closing_circle")
                {
                    double radius = Scalar(Literal(call.ControlIn[0], constants, call.Line), "Radius", call.Line);
                    if (radius < 0)
                        throw new ExportException("半徑不能是負的", call.Line);
                    string method = bare switch
                    {
                        "dilation_circle" => "DilateCircle",
                        "erosion_circle" => "ErodeCircle",
                        "opening_circle" => "OpenCircle",
                        _ => "CloseCircle",
                    };
                    expr = "Operators." + method + "(" + Ident(sourceName) + ", " + CsDouble(radius) + ")";
                }
                else if (bare == "boundary")
                {
                    string kind = Shown(Literal(call.ControlIn[0], constants, call.Line)).Trim().Trim('\'', '"').ToLowerInvariant();
                    if (kind is not ("inner" or "outer"))
                        throw new ExportException("BoundaryType 只能是 inner 或 outer", call.Line);
                    expr = "Operators.Boundary(" + Ident(sourceName) + ", " + CsString(kind) + ")";
                }
                else
                {
                    int wide = Whole(Literal(call.ControlIn[0], constants, call.Line), "Width", call.Line);
                    int high = Whole(Literal(call.ControlIn[1], constants, call.Line), "Height", call.Line);
                    string method = bare switch
                    {
                        "dilation_rectangle1" => "DilateRectangle",
                        "erosion_rectangle1" => "ErodeRectangle",
                        "opening_rectangle1" => "OpenRectangle",
                        _ => "CloseRectangle",
                    };
                    expr = "Operators." + method + "(" + Ident(sourceName) + ", " + wide + ", " + high + ")";
                }
                Assign(name, expr, "region", call.Line);
                regions[name] = "one";
                images.Remove(name);
                lastRegion = name;
                continue;
            }
            if (bare is "union2" or "intersection" or "difference" or "concat_obj")
            {
                string left = call.IconicIn[0];
                string right = call.IconicIn[1];
                if (!regions.ContainsKey(left) || !regions.ContainsKey(right))
                {
                    string missing = regions.ContainsKey(left) ? right : left;
                    throw new ExportException("找不到區域 " + missing, call.Line);
                }
                string name = call.IconicOut[0];
                string method = bare switch
                {
                    "union2" => "Union",
                    "intersection" => "Intersection",
                    "difference" => "Difference",
                    _ => "Concat",
                };
                Assign(name, "Operators." + method + "(" + Ident(left) + ", " + Ident(right) + ")", bare == "concat_obj" ? "regions" : "region", call.Line);
                regions[name] = bare == "concat_obj" ? "many" : "one";
                images.Remove(name);
                lastRegion = name;
                continue;
            }
            if (bare == "select_obj")
            {
                string sourceName = call.IconicIn[0];
                if (!regions.TryGetValue(sourceName, out string? shape))
                    throw new ExportException("找不到區域 " + sourceName, call.Line);
                Value index = Literal(call.ControlIn[0], constants, call.Line);
                string name = call.IconicOut[0];
                if (index is ListValue list)
                {
                    string listed = string.Join(", ", list.Items.Select(Rounded));
                    string source = shape == "many" ? Ident(sourceName) : "new[] { " + Ident(sourceName) + " }";
                    Assign(name, "Operators.Select(" + source + ", new[] { " + listed + " })", "regions", call.Line);
                    regions[name] = "many";
                }
                else
                {
                    Assign(name, "Operators.Select(" + Ident(sourceName) + ", " + Rounded(index) + ")", "region", call.Line);
                    regions[name] = "one";
                }
                images.Remove(name);
                lastRegion = name;
                continue;
            }
            if (bare == "select_shape")
            {
                string sourceName = call.IconicIn[0];
                if (!regions.ContainsKey(sourceName))
                    throw new ExportException("找不到區域 " + sourceName, call.Line);
                var features = AsStrings(Literal(call.ControlIn[0], constants, call.Line))
                    .Select(item => item.Trim().Trim('\'', '"').ToLowerInvariant())
                    .ToList();
                string? unknown = features.FirstOrDefault(item => !Shape.Contains(item));
                if (unknown != null)
                    throw new ExportException("匯出的 select_shape 還沒有做特徵 " + unknown, call.Line);
                string operation = Shown(Literal(call.ControlIn[1], constants, call.Line)).Trim().Trim('\'', '"').ToLowerInvariant();
                if (operation is not ("and" or "or"))
                    throw new ExportException("Operation 只能是 and 或 or", call.Line);
                var lows = AsNumbers(Literal(call.ControlIn[2], constants, call.Line), call.Line);
                var highs = AsNumbers(Literal(call.ControlIn[3], constants, call.Line), call.Line);
                if (lows.Count == 1 && features.Count > 1)
                    lows = Enumerable.Repeat(lows[0], features.Count).ToList();
                if (highs.Count == 1 && features.Count > 1)
                    highs = Enumerable.Repeat(highs[0], features.Count).ToList();
                if (lows.Count != features.Count || highs.Count != features.Count)
                    throw new ExportException("Min 和 Max 的長度要和特徵一樣", call.Line);
                string name = call.IconicOut[0];
                string names = string.Join(", ", features.Select(CsString));
                string lowText = string.Join(", ", lows.Select(CsDouble));
                string highText = string.Join(", ", highs.Select(CsDouble));
                Assign(
                    name,
                    "Operators.SelectShape(" + Ident(sourceName) + ", new[] { " + names + " }, " + CsString(operation) + ", new[] { " + lowText + " }, new[] { " + highText + " })",
                    "regions",
                    call.Line);
                regions[name] = "many";
                images.Remove(name);
                lastRegion = name;
                continue;
            }
            if (bare == "gen_rectangle1")
            {
                int[] coords = Coords(call.ControlIn, constants, call.Line);
                string name = call.IconicOut[0];
                string joined = string.Join(", ", coords);
                string expr = lastImage.Length > 0
                    ? "Operators.Rectangle(" + Ident(lastImage) + ", " + joined + ")"
                    : "Operators.Rectangle(" + joined + ")";
                Assign(name, expr, "region", call.Line);
                regions[name] = "one";
                images.Remove(name);
                lastRegion = name;
                continue;
            }
            if (bare == "get_image_size")
            {
                string sourceName = call.IconicIn[0];
                if (!images.Contains(sourceName))
                    throw new ExportException("找不到影像 " + sourceName, call.Line);
                Assign(call.ControlOut[0], Ident(sourceName) + ".Width", "number", call.Line);
                Assign(call.ControlOut[1], Ident(sourceName) + ".Height", "number", call.Line);
                continue;
            }
            if (bare == "get_grayval")
            {
                string sourceName = call.IconicIn[0];
                if (!images.Contains(sourceName))
                    throw new ExportException("找不到影像 " + sourceName, call.Line);
                Value row = Literal(call.ControlIn[0], constants, call.Line);
                Value column = Literal(call.ControlIn[1], constants, call.Line);
                if (row is ListValue || column is ListValue)
                    throw new ExportException("匯出的 get_grayval 一次讀一個點", call.Line);
                Assign(call.ControlOut[0], "Operators.GrayAt(" + Ident(sourceName) + ", " + Rounded(row) + ", " + Rounded(column) + ")", "intarray", call.Line);
                continue;
            }
            if (bare == "test_region_point")
            {
                string sourceName = call.IconicIn[0];
                if (!regions.ContainsKey(sourceName))
                    throw new ExportException("找不到區域 " + sourceName, call.Line);
                Value row = Literal(call.ControlIn[0], constants, call.Line);
                Value column = Literal(call.ControlIn[1], constants, call.Line);
                if (row is ListValue || column is ListValue)
                    throw new ExportException("匯出的 test_region_point 一次測一個點", call.Line);
                Assign(call.ControlOut[0], "Operators.Contains(" + Ident(sourceName) + ", " + Rounded(row) + ", " + Rounded(column) + ")", "number", call.Line);
                continue;
            }
            if (bare == "smallest_rectangle1")
            {
                string sourceName = call.IconicIn[0];
                if (!regions.ContainsKey(sourceName))
                    throw new ExportException("找不到區域 " + sourceName, call.Line);
                measureIndex++;
                string temp = "exportBounds" + measureIndex.ToString(CultureInfo.InvariantCulture);
                body.Add("        var " + temp + " = Operators.Bounds(" + Ident(sourceName) + ");");
                string row1 = call.ControlOut[0];
                string column1 = call.ControlOut[1];
                string row2 = call.ControlOut[2];
                string column2 = call.ControlOut[3];
                Assign(row1, "new int[" + temp + ".Length]", "intarray", call.Line);
                Assign(column1, "new int[" + temp + ".Length]", "intarray", call.Line);
                Assign(row2, "new int[" + temp + ".Length]", "intarray", call.Line);
                Assign(column2, "new int[" + temp + ".Length]", "intarray", call.Line);
                body.Add("        for (int exportIndex = 0; exportIndex < " + temp + ".Length; exportIndex++)");
                body.Add("        {");
                body.Add("            " + Ident(row1) + "[exportIndex] = " + temp + "[exportIndex].Row1;");
                body.Add("            " + Ident(column1) + "[exportIndex] = " + temp + "[exportIndex].Column1;");
                body.Add("            " + Ident(row2) + "[exportIndex] = " + temp + "[exportIndex].Row2;");
                body.Add("            " + Ident(column2) + "[exportIndex] = " + temp + "[exportIndex].Column2;");
                body.Add("        }");
                continue;
            }
            if (bare == "intensity")
            {
                string regionName = call.IconicIn[0];
                string imageName = call.IconicIn[1];
                if (!regions.ContainsKey(regionName))
                    throw new ExportException("找不到區域 " + regionName, call.Line);
                if (!images.Contains(imageName))
                    throw new ExportException("找不到影像 " + imageName, call.Line);
                measureIndex++;
                string temp = "exportTone" + measureIndex.ToString(CultureInfo.InvariantCulture);
                body.Add("        var " + temp + " = Operators.IntensityOf(" + Ident(regionName) + ", " + Ident(imageName) + ");");
                string meanName = call.ControlOut[0];
                string deviationName = call.ControlOut[1];
                Assign(meanName, "new double[" + temp + ".Length]", "numbers", call.Line);
                Assign(deviationName, "new double[" + temp + ".Length]", "numbers", call.Line);
                body.Add("        for (int exportIndex = 0; exportIndex < " + temp + ".Length; exportIndex++)");
                body.Add("        {");
                body.Add("            " + Ident(meanName) + "[exportIndex] = " + temp + "[exportIndex].Mean;");
                body.Add("            " + Ident(deviationName) + "[exportIndex] = " + temp + "[exportIndex].Deviation;");
                body.Add("        }");
                continue;
            }
            if (bare == "count_obj")
            {
                string sourceName = call.IconicIn[0];
                if (!regions.ContainsKey(sourceName))
                    throw new ExportException("找不到區域 " + sourceName, call.Line);
                Assign(call.ControlOut[0], "Operators.Count(" + Ident(sourceName) + ")", "number", call.Line);
                continue;
            }
            if (bare == "area_center")
            {
                string sourceName = call.IconicIn[0];
                if (!regions.ContainsKey(sourceName))
                    throw new ExportException("找不到區域 " + sourceName, call.Line);
                measureIndex++;
                string temp = "exportMeasured" + measureIndex.ToString(CultureInfo.InvariantCulture);
                body.Add("        var " + temp + " = Operators.AreaCenterOf(" + Ident(sourceName) + ");");
                string areaName = call.ControlOut[0];
                string rowName = call.ControlOut[1];
                string columnName = call.ControlOut[2];
                Assign(areaName, "new int[" + temp + ".Length]", "numbers", call.Line);
                Assign(rowName, "new double[" + temp + ".Length]", "numbers", call.Line);
                Assign(columnName, "new double[" + temp + ".Length]", "numbers", call.Line);
                body.Add("        for (int exportIndex = 0; exportIndex < " + temp + ".Length; exportIndex++)");
                body.Add("        {");
                body.Add("            " + Ident(areaName) + "[exportIndex] = " + temp + "[exportIndex].Area;");
                body.Add("            " + Ident(rowName) + "[exportIndex] = " + temp + "[exportIndex].Row;");
                body.Add("            " + Ident(columnName) + "[exportIndex] = " + temp + "[exportIndex].Column;");
                body.Add("        }");
                continue;
            }
            string target = call.IconicIn[0];
            displayed = true;
            if (images.Contains(target))
            {
                shownImage = target;
                shownRegion = "";
                continue;
            }
            if (regions.ContainsKey(target))
            {
                if (shownImage.Length == 0)
                    shownImage = lastImage;
                shownRegion = target;
                continue;
            }
            throw new ExportException("找不到要顯示的 " + target, call.Line);
        }

        if (!displayed)
        {
            shownImage = lastImage;
            shownRegion = lastRegion;
        }
        if (shownImage.Length == 0)
            throw new ExportException("沒有影像可以顯示");
        string overlay = shownRegion.Length > 0
            ? "Operators.Overlays(" + Ident(shownRegion) + ")"
            : "System.Array.Empty<Overlay>()";
        string image = Ident(shownImage);
        body.Add("        view(" + image + ".Width, " + image + ".Height, " + image + ".ForDisplay(), " + image + ".Layout, " + overlay + ");");
        return "using System;\nusing Vizion;\n\n// WinForms：using Vizion.Forms; 然後 " + className + ".Run(window.Show);\npublic static class " + className + "\n{\n    public static void Run(ViewShow view)\n    {\n" + string.Join("\n", body) + "\n    }\n}\n";
    }

    static int[] Coords(string[] exprs, Dictionary<string, Value> constants, int line)
    {
        string[] labels = ["Row1", "Column1", "Row2", "Column2"];
        var coords = new int[4];
        for (int i = 0; i < 4; i++)
            coords[i] = Whole(Literal(exprs[i], constants, line), labels[i], line);
        return coords;
    }

    static List<Stmt> Parse(string[] rawLines, Dictionary<string, OperatorDoc> ops)
    {
        var stmts = new List<Stmt>();
        for (int index = 0; index < rawLines.Length; index++)
        {
            string text = rawLines[index];
            int line = index + 1;
            while (text.TrimEnd().EndsWith('\\'))
            {
                text = text.TrimEnd()[..^1] + " ";
                index++;
                if (index >= rawLines.Length)
                    break;
                text += rawLines[index];
            }
            string raw = text.Trim();
            if (raw.EndsWith(';'))
                raw = raw[..^1].Trim();
            if (raw.Length == 0 || raw.StartsWith('*') || raw.StartsWith("//"))
                continue;
            string low = raw.ToLowerInvariant();
            if (low is "endif" or "else" or "endfor" or "endwhile" or "endprocedure" or "endproc" or "try" or "endtry")
            {
                stmts.Add(new OtherStmt(line));
                continue;
            }
            if (low == "catch" || low.StartsWith("catch ", StringComparison.Ordinal) || low.StartsWith("catch(", StringComparison.Ordinal))
            {
                stmts.Add(new OtherStmt(line));
                continue;
            }
            if (low.StartsWith("procedure ", StringComparison.Ordinal) || low.StartsWith("procedure(", StringComparison.Ordinal))
            {
                stmts.Add(new OtherStmt(line));
                continue;
            }
            if (low.StartsWith("if ", StringComparison.Ordinal) || low.StartsWith("if(", StringComparison.Ordinal))
            {
                if (!IfPattern.IsMatch(raw))
                    throw new ExportException("if 要寫成 if (條件)", line);
                stmts.Add(new OtherStmt(line));
                continue;
            }
            if (low.StartsWith("while ", StringComparison.Ordinal) || low.StartsWith("while(", StringComparison.Ordinal))
            {
                if (!WhilePattern.IsMatch(raw))
                    throw new ExportException("while 要寫成 while (條件)", line);
                stmts.Add(new OtherStmt(line));
                continue;
            }
            if (low.StartsWith("for ", StringComparison.Ordinal) || low.StartsWith("for\t", StringComparison.Ordinal))
            {
                stmts.Add(new OtherStmt(line));
                continue;
            }
            Match assign = AssignPattern.Match(raw);
            if (assign.Success)
            {
                stmts.Add(new AssignStmt(line, assign.Groups[1].Value, assign.Groups[2].Value.Trim()));
                continue;
            }
            Match vision = VisionCallPattern.Match(raw);
            if (vision.Success)
            {
                stmts.Add(ParseCall(vision.Groups[1].Value, vision.Groups[2].Value, line, ops));
                continue;
            }
            Match bare = BareCallPattern.Match(raw);
            if (bare.Success)
            {
                string key = bare.Groups[1].Value;
                if (ops.ContainsKey(key))
                    throw new ExportException("算子要寫成 " + ops[key].Name + " (...)", line);
                throw new ExportException("未知算子 " + key, line);
            }
            throw new ExportException("看不懂這一行。註解用 *，指定用 :=，呼叫用 Vizion.operator (...)", line);
        }
        return stmts;
    }

    static CallStmt ParseCall(string name, string args, int line, Dictionary<string, OperatorDoc> ops)
    {
        int dot = name.IndexOf('.');
        string bare = (dot >= 0 ? name[(dot + 1)..] : name).ToLowerInvariant();
        if (!ops.TryGetValue(bare, out OperatorDoc? op))
            throw new ExportException("未知算子 " + name, line);
        var iconicIn = new List<OperatorParam>();
        var iconicOut = new List<OperatorParam>();
        var controlIn = new List<OperatorParam>();
        var controlOut = new List<OperatorParam>();
        foreach (OperatorParam param in op.Params)
        {
            switch (param.Role)
            {
                case "圖像輸入": iconicIn.Add(param); break;
                case "圖像輸出": iconicOut.Add(param); break;
                case "控制輸入": controlIn.Add(param); break;
                case "控制輸出": controlOut.Add(param); break;
            }
        }
        string[] tokens = SplitCommas(args);
        int optional = 0;
        for (int i = controlIn.Count - 1; i >= 0; i--)
        {
            if (controlIn[i].Default == null)
                break;
            optional++;
        }
        int minimum = iconicIn.Count + iconicOut.Count + (controlIn.Count - optional) + controlOut.Count;
        int maximum = iconicIn.Count + iconicOut.Count + controlIn.Count + controlOut.Count;
        if (tokens.Length < minimum || tokens.Length > maximum)
        {
            string span = minimum == maximum ? minimum.ToString(CultureInfo.InvariantCulture) : minimum + " 到 " + maximum;
            throw new ExportException(op.Name + " 需要 " + span + " 個參數，目前是 " + tokens.Length + " 個。範本：" + op.Example, line);
        }
        var inArgs = tokens.Take(iconicIn.Count).ToArray();
        var outArgs = tokens.Skip(iconicIn.Count).Take(iconicOut.Count).ToArray();
        string[] tail = tokens.Skip(iconicIn.Count + iconicOut.Count).ToArray();
        string[] coutArgs;
        string[] cinArgs;
        if (controlOut.Count > 0)
        {
            coutArgs = tail.Skip(tail.Length - controlOut.Count).ToArray();
            cinArgs = tail.Take(tail.Length - controlOut.Count).ToArray();
        }
        else
        {
            coutArgs = [];
            cinArgs = tail;
        }
        var controls = cinArgs.ToList();
        while (controls.Count < controlIn.Count)
        {
            string? fallback = controlIn[controls.Count].Default;
            if (fallback == null)
                throw new ExportException("參數不夠", line);
            controls.Add(fallback);
        }
        foreach (string token in inArgs.Concat(outArgs).Concat(coutArgs))
        {
            if (!IdentPattern.IsMatch(token))
                throw new ExportException("輸出與圖像輸入必須是變數名稱，得到 " + token, line);
        }
        return new CallStmt(line, op.Name, bare, inArgs, outArgs, controls.ToArray(), coutArgs);
    }

    static string[] SplitCommas(string text)
    {
        string raw = text.Trim();
        if (raw.Length == 0)
            return [];
        var parts = new List<string>();
        var buf = new System.Text.StringBuilder();
        char quote = '\0';
        int depth = 0;
        foreach (char ch in raw)
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
        parts.Add(buf.ToString().Trim());
        return parts.ToArray();
    }

    static string Ident(string name) => Keywords.Contains(name) ? "@" + name : name;

    static string CsString(string value)
    {
        string escaped = value.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\r", "\\r").Replace("\n", "\\n");
        return "\"" + escaped + "\"";
    }

    static string CsNumber(double number)
    {
        if (!double.IsNaN(number) && !double.IsInfinity(number) && number == Math.Truncate(number) && Math.Abs(number) < 1e15)
            return ((long)number).ToString(CultureInfo.InvariantCulture);
        string text = number.ToString("G17", CultureInfo.InvariantCulture);
        if (!text.Contains('e') && !text.Contains('E') && !text.Contains('.'))
            text += ".0";
        return text;
    }

    static string CsDouble(double value)
    {
        string text = CsNumber(value);
        if (!text.Contains('.') && !text.Contains('e') && !text.Contains('E'))
            text += ".0";
        return text;
    }

    static Value Literal(string expr, Dictionary<string, Value> variables, int line)
    {
        try
        {
            return EvalExpr(expr, variables);
        }
        catch (ExprException ex)
        {
            throw new ExportException(ex.Message, line);
        }
    }

    static double Scalar(Value value, string name, int line)
    {
        if (value is ListValue list)
        {
            if (list.Items.Count != 1)
                throw new ExportException(name + " 需要單一數值，得到 " + list.Items.Count + " 個", line);
            value = list.Items[0];
        }
        if (value is NumValue number)
            return number.Number;
        throw new ExportException(name + " 不是數值：" + Shown(value), line);
    }

    static int Whole(Value value, string name, int line) =>
        (int)Math.Round(Scalar(value, name, line), MidpointRounding.ToEven);

    static int Rounded(Value value)
    {
        if (value is not NumValue number)
            throw new ExprException("需要數值，得到 " + Shown(value));
        return (int)Math.Round(number.Number, MidpointRounding.ToEven);
    }

    static List<string> AsStrings(Value value)
    {
        if (value is StrValue text)
            return [text.Text];
        if (value is ListValue list)
            return list.Items.Select(Shown).ToList();
        return [Shown(value)];
    }

    static List<double> AsNumbers(Value value, int line)
    {
        if (value is NumValue one)
            return [one.Number];
        if (value is not ListValue list)
            throw new ExportException("不是數值", line);
        var nums = new List<double>();
        foreach (Value item in list.Items)
        {
            if (item is not NumValue number)
                throw new ExportException("不是數值", line);
            nums.Add(number.Number);
        }
        return nums;
    }

    static string Shown(Value value) => value switch
    {
        StrValue text => text.Text,
        NumValue number => CsNumber(number.Number),
        ListValue list => "[" + string.Join(", ", list.Items.Select(Shown)) + "]",
        _ => "",
    };

    abstract record Stmt(int Line);
    sealed record AssignStmt(int Line, string Name, string Expr) : Stmt(Line);
    sealed record CallStmt(int Line, string Name, string Bare, string[] IconicIn, string[] IconicOut, string[] ControlIn, string[] ControlOut) : Stmt(Line);
    sealed record OtherStmt(int Line) : Stmt(Line);

    static Value EvalExpr(string expr, Dictionary<string, Value> variables)
    {
        string source = Rewrite(expr.Trim());
        if (source.Length == 0)
            throw new ExprException("空運算式");
        var reader = new Reader(source);
        Value value = ParseOr(reader, variables, expr);
        if (!reader.End)
            throw new ExprException("運算式語法錯誤：" + expr);
        return value;
    }

    static string Rewrite(string expr)
    {
        var chars = new System.Text.StringBuilder();
        int i = 0;
        while (i < expr.Length)
        {
            char ch = expr[i];
            if (ch is '\'' or '"')
            {
                int j = i + 1;
                while (j < expr.Length && expr[j] != ch)
                    j++;
                int end = j < expr.Length ? j : expr.Length - 1;
                chars.Append(expr, i, end - i + 1);
                i = j + 1;
                continue;
            }
            if (ch == '|')
            {
                int j = i + 1;
                while (j < expr.Length && (char.IsLetterOrDigit(expr[j]) || expr[j] == '_'))
                    j++;
                if (j < expr.Length && expr[j] == '|' && j > i + 1)
                {
                    chars.Append("__len__(").Append(expr, i + 1, j - i - 1).Append(')');
                    i = j + 1;
                    continue;
                }
            }
            if (ch == '<' && i + 1 < expr.Length && expr[i + 1] == '>')
            {
                chars.Append("!=");
                i += 2;
                continue;
            }
            if (ch == '#')
            {
                chars.Append("!=");
                i++;
                continue;
            }
            if (ch == '=' && (i == 0 || expr[i - 1] is not ('<' or '>' or '!' or '=')) && (i + 1 >= expr.Length || expr[i + 1] != '='))
            {
                chars.Append("==");
                i++;
                continue;
            }
            chars.Append(ch);
            i++;
        }
        return chars.ToString();
    }

    static Value ParseOr(Reader reader, Dictionary<string, Value> variables, string origin)
    {
        Value left = ParseAnd(reader, variables, origin);
        while (reader.Take("or"))
        {
            Value right = ParseAnd(reader, variables, origin);
            left = new NumValue(Truthy(left) || Truthy(right) ? 1 : 0);
        }
        return left;
    }

    static Value ParseAnd(Reader reader, Dictionary<string, Value> variables, string origin)
    {
        Value left = ParseNot(reader, variables, origin);
        while (reader.Take("and"))
        {
            Value right = ParseNot(reader, variables, origin);
            left = new NumValue(Truthy(left) && Truthy(right) ? 1 : 0);
        }
        return left;
    }

    static Value ParseNot(Reader reader, Dictionary<string, Value> variables, string origin)
    {
        if (!reader.Take("not"))
            return ParseCmp(reader, variables, origin);
        return new NumValue(Truthy(ParseNot(reader, variables, origin)) ? 0 : 1);
    }

    static Value ParseCmp(Reader reader, Dictionary<string, Value> variables, string origin)
    {
        Value left = ParseAdd(reader, variables, origin);
        if (reader.Kind is not ("==" or "!=" or "<" or ">" or "<=" or ">="))
            return left;
        string op = reader.Kind;
        reader.Next();
        if (reader.Kind is "==" or "!=" or "<" or ">" or "<=" or ">=")
            throw new ExprException("暫不支援連續比較");
        Value right = ParseAdd(reader, variables, origin);
        return new NumValue(Compare(op, left, right) ? 1 : 0);
    }

    static Value ParseAdd(Reader reader, Dictionary<string, Value> variables, string origin)
    {
        Value left = ParseMul(reader, variables, origin);
        while (reader.Kind is "+" or "-")
        {
            string op = reader.Kind;
            reader.Next();
            Value right = ParseMul(reader, variables, origin);
            if (op == "+" && (left is StrValue || right is StrValue))
            {
                if (left is ListValue || right is ListValue)
                    throw new ExprException("字串不能和 tuple 相加");
                left = new StrValue(Shown(left) + Shown(right));
                continue;
            }
            left = new NumValue(op == "+" ? Num(left) + Num(right) : Num(left) - Num(right));
        }
        return left;
    }

    static Value ParseMul(Reader reader, Dictionary<string, Value> variables, string origin)
    {
        Value left = ParseUnary(reader, variables, origin);
        while (reader.Kind is "*" or "/" or "//" or "%")
        {
            string op = reader.Kind;
            reader.Next();
            Value right = ParseUnary(reader, variables, origin);
            double a = Num(left);
            double b = Num(right);
            left = new NumValue(op switch
            {
                "*" => a * b,
                "/" => a / b,
                "//" => Math.Floor(a / b),
                _ => b == 0 ? throw new ExprException("不支援的運算") : a - Math.Floor(a / b) * b,
            });
        }
        return left;
    }

    static Value ParseUnary(Reader reader, Dictionary<string, Value> variables, string origin)
    {
        if (reader.Take("+"))
            return ParseUnary(reader, variables, origin);
        if (reader.Take("-"))
            return new NumValue(-Num(ParseUnary(reader, variables, origin)));
        return ParsePow(reader, variables, origin);
    }

    static Value ParsePow(Reader reader, Dictionary<string, Value> variables, string origin)
    {
        Value left = ParsePrimary(reader, variables, origin);
        if (!reader.Take("**"))
            return left;
        return new NumValue(Math.Pow(Num(left), Num(ParseUnary(reader, variables, origin))));
    }

    static Value ParsePrimary(Reader reader, Dictionary<string, Value> variables, string origin)
    {
        if (reader.Take("("))
        {
            Value inner = ParseOr(reader, variables, origin);
            if (!reader.Take(")"))
                throw new ExprException("運算式語法錯誤：" + origin);
            return Subscript(inner, "", reader, variables, origin);
        }
        if (reader.Take("["))
        {
            var items = new List<Value>();
            if (!reader.Is("]"))
            {
                items.Add(ParseOr(reader, variables, origin));
                while (reader.Take(","))
                {
                    if (reader.Is("]"))
                        break;
                    items.Add(ParseOr(reader, variables, origin));
                }
            }
            if (!reader.Take("]"))
                throw new ExprException("運算式語法錯誤：" + origin);
            return new ListValue(items);
        }
        if (reader.Kind == "number")
        {
            var value = new NumValue(reader.Number);
            reader.Next();
            return value;
        }
        if (reader.Kind == "string")
        {
            var value = new StrValue(reader.TextValue);
            reader.Next();
            return value;
        }
        if (reader.Kind == "ident")
        {
            string name = reader.TextValue;
            reader.Next();
            if (reader.Take("("))
            {
                var args = new List<Value>();
                if (!reader.Is(")"))
                {
                    args.Add(ParseOr(reader, variables, origin));
                    while (reader.Take(","))
                        args.Add(ParseOr(reader, variables, origin));
                }
                if (!reader.Take(")"))
                    throw new ExprException("運算式語法錯誤：" + origin);
                return Call(name, args);
            }
            return Subscript(Lookup(name, variables), name, reader, variables, origin);
        }
        throw new ExprException("運算式語法錯誤：" + origin);
    }

    static Value Subscript(Value value, string name, Reader reader, Dictionary<string, Value> variables, string origin)
    {
        while (reader.Take("["))
        {
            Value index = ParseOr(reader, variables, origin);
            if (!reader.Take("]"))
                throw new ExprException("運算式語法錯誤：" + origin);
            if (value is not ListValue list)
                throw new ExprException("不支援的運算式");
            int originIndex = name.StartsWith("hv_", StringComparison.OrdinalIgnoreCase) ? 0 : 1;
            int pos = (int)Math.Round(Num(index), MidpointRounding.ToEven);
            int slot = originIndex == 0 ? pos : pos - 1;
            if (slot < 0 || slot >= list.Items.Count)
                throw new ExprException("索引 " + pos + " 超出範圍");
            value = list.Items[slot];
            name = "";
        }
        return value;
    }

    static Value Lookup(string name, Dictionary<string, Value> variables)
    {
        if (name is "true" or "True")
            return new NumValue(1);
        if (name is "false" or "False")
            return new NumValue(0);
        if (!variables.TryGetValue(name, out Value? value))
            throw new ExprException("未知變數 " + name);
        return value;
    }

    static Value Call(string name, List<Value> args)
    {
        if (name == "__len__")
        {
            if (args.Count != 1)
                throw new ExprException("長度語法錯誤");
            return new NumValue(args[0] is ListValue list ? list.Items.Count : 1);
        }
        if (name is "min" or "max" or "sum" or "mean")
        {
            var nums = Flat(args);
            if (nums.Count == 0)
                throw new ExprException("空 tuple");
            double result = name switch
            {
                "min" => nums.Min(),
                "max" => nums.Max(),
                "sum" => nums.Sum(),
                _ => nums.Sum() / nums.Count,
            };
            return new NumValue(result);
        }
        if (name is "abs" or "sqrt" or "int" or "rad" or "deg")
        {
            if (args.Count != 1)
                throw new ExprException("未知函式 " + name + "。可用 min、max、sum、mean、abs、sqrt、round、rad、deg");
            double number = Num(args[0]);
            double result = name switch
            {
                "abs" => Math.Abs(number),
                "sqrt" => Math.Sqrt(number),
                "int" => Math.Truncate(number),
                "rad" => number * Math.PI / 180.0,
                _ => number * 180.0 / Math.PI,
            };
            return new NumValue(result);
        }
        if (name == "round")
        {
            if (args.Count is not (1 or 2))
                throw new ExprException("round 需要 1 或 2 個參數");
            int digits = args.Count == 1 ? 0 : (int)Num(args[1]);
            double number = Math.Round(Num(args[0]), digits, MidpointRounding.AwayFromZero);
            return new NumValue(number);
        }
        throw new ExprException("未知函式 " + name + "。可用 min、max、sum、mean、abs、sqrt、round、rad、deg");
    }

    static List<double> Flat(List<Value> args)
    {
        var nums = new List<double>();
        foreach (Value arg in args)
        {
            if (arg is ListValue list)
            {
                foreach (Value item in list.Items)
                    nums.Add(Num(item));
            }
            else
                nums.Add(Num(arg));
        }
        return nums;
    }

    static bool Compare(string op, Value left, Value right)
    {
        if (left is StrValue || right is StrValue)
        {
            string a = Shown(left);
            string b = Shown(right);
            int order = string.CompareOrdinal(a, b);
            return op switch
            {
                "==" => order == 0,
                "!=" => order != 0,
                "<" => order < 0,
                ">" => order > 0,
                "<=" => order <= 0,
                _ => order >= 0,
            };
        }
        double aNum = Num(left);
        double bNum = Num(right);
        return op switch
        {
            "==" => aNum == bNum,
            "!=" => aNum != bNum,
            "<" => aNum < bNum,
            ">" => aNum > bNum,
            "<=" => aNum <= bNum,
            _ => aNum >= bNum,
        };
    }

    static bool Truthy(Value value)
    {
        if (value is ListValue list)
            return list.Items.Count > 0 && list.Items.All(Truthy);
        if (value is StrValue text)
            return text.Text.Length > 0;
        if (value is NumValue number)
            return number.Number != 0;
        return false;
    }

    static double Num(Value value)
    {
        if (value is NumValue number)
            return number.Number;
        throw new ExprException("需要數值，得到 " + Shown(value));
    }

    sealed class Reader
    {
        readonly string Text;
        int Index;
        public string Kind = "end";
        public string TextValue = "";
        public double Number;

        public Reader(string text)
        {
            Text = text;
            Next();
        }

        public bool End => Kind == "end";

        public bool Is(string kind) => Kind == kind;

        public bool Take(string kind)
        {
            if (Kind != kind)
                return false;
            Next();
            return true;
        }

        public void Next()
        {
            while (Index < Text.Length && char.IsWhiteSpace(Text[Index]))
                Index++;
            if (Index >= Text.Length)
            {
                Kind = "end";
                TextValue = "";
                return;
            }
            char ch = Text[Index];
            if (ch is '\'' or '"')
            {
                int j = Index + 1;
                while (j < Text.Length && Text[j] != ch)
                    j++;
                TextValue = Text[(Index + 1)..(j < Text.Length ? j : Text.Length)];
                Index = j < Text.Length ? j + 1 : Text.Length;
                Kind = "string";
                return;
            }
            if (char.IsDigit(ch) || (ch == '.' && Index + 1 < Text.Length && char.IsDigit(Text[Index + 1])))
            {
                int j = Index + 1;
                while (j < Text.Length && (char.IsDigit(Text[j]) || Text[j] == '.'))
                    j++;
                if (j < Text.Length && Text[j] is 'e' or 'E')
                {
                    int k = j + 1;
                    if (k < Text.Length && Text[k] is '+' or '-')
                        k++;
                    int mark = k;
                    while (k < Text.Length && char.IsDigit(Text[k]))
                        k++;
                    if (k > mark)
                        j = k;
                }
                TextValue = Text[Index..j];
                Number = double.Parse(TextValue, CultureInfo.InvariantCulture);
                Index = j;
                Kind = "number";
                return;
            }
            if (char.IsLetter(ch) || ch == '_')
            {
                int j = Index + 1;
                while (j < Text.Length && (char.IsLetterOrDigit(Text[j]) || Text[j] == '_'))
                    j++;
                TextValue = Text[Index..j];
                Index = j;
                Kind = TextValue is "and" or "or" or "not" ? TextValue : "ident";
                return;
            }
            if (Index + 1 < Text.Length)
            {
                string two = Text.Substring(Index, 2);
                if (two is "==" or "!=" or "<=" or ">=" or "**" or "//")
                {
                    Kind = two;
                    Index += 2;
                    return;
                }
            }
            Kind = ch.ToString();
            Index++;
        }
    }

    abstract record Value;
    sealed record NumValue(double Number) : Value;
    sealed record StrValue(string Text) : Value;
    sealed record ListValue(List<Value> Items) : Value;
}

sealed class ExprException : Exception
{
    public ExprException(string message) : base(message)
    {
    }
}
