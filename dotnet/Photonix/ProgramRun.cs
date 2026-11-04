using System.Diagnostics;
using System.Globalization;
using System.IO;
using Vizion;

namespace Photonix;

/// <summary>一行一個敘述。能做的算子呼叫 Vizion.Operators，做不到就停在那一行。</summary>
sealed class ProgramRun
{
    public Dictionary<string, object> Vars { get; } = new(StringComparer.Ordinal);
    public string? ProjectFolder { get; set; }
    public int Pc { get; set; } = 1;
    public bool Done { get; private set; }
    public string? Error { get; private set; }
    public int ErrorLine { get; private set; }
    public bool Cancel { get; set; }
    public int BreakSkip { get; set; }
    public int Steps { get; private set; }
    public Dictionary<int, (double Ms, int Count)> Times { get; } = new();
    public ImageBuffer? Image { get; private set; }
    public Overlay[] Shapes { get; private set; } = [];
    public List<string> Messages { get; } = [];

    public void Reset()
    {
        Vars.Clear();
        Pc = 1;
        Done = false;
        Error = null;
        ErrorLine = 0;
        Cancel = false;
        BreakSkip = 0;
        Steps = 0;
        Times.Clear();
        Image = null;
        Shapes = [];
        Messages.Clear();
    }

    public void MoveTo(int line)
    {
        Error = null;
        ErrorLine = 0;
        Done = false;
        BreakSkip = 0;
        Cancel = false;
        Pc = Math.Max(1, line);
    }

    public Halt Step(string[] lines, bool trace)
    {
        if (Error != null)
            return Halt.Error;
        if (Done)
            return Halt.Done;
        Halt arrived = Arrive(lines, breakpoints: null);
        if (arrived != Halt.None)
            return arrived;
        return ExecuteOne(lines, trace);
    }

    public Halt Run(string[] lines, IReadOnlySet<int> breakpoints, bool trace, int maxSteps)
    {
        int taken = 0;
        while (taken < maxSteps)
        {
            if (Cancel)
                return Halt.Stopped;
            if (Error != null)
                return Halt.Error;
            if (Done)
                return Halt.Done;
            if (Pc >= 1 && Pc <= lines.Length && breakpoints.Contains(Pc) && BreakSkip != Pc)
            {
                BreakSkip = Pc;
                return Halt.Breakpoint;
            }
            Halt arrived = Arrive(lines, breakpoints);
            if (arrived != Halt.None)
                return arrived;
            Halt one = ExecuteOne(lines, trace);
            taken++;
            if (one != Halt.None)
                return one;
        }
        return Halt.Yield;
    }

    Halt Arrive(string[] lines, IReadOnlySet<int>? breakpoints)
    {
        while (Pc <= lines.Length && IsSkippable(lines[Pc - 1]))
        {
            if (breakpoints != null && breakpoints.Contains(Pc) && BreakSkip != Pc)
            {
                BreakSkip = Pc;
                return Halt.Breakpoint;
            }
            Pc++;
        }
        if (Pc > lines.Length)
        {
            Done = true;
            return Halt.Done;
        }
        if (breakpoints != null && breakpoints.Contains(Pc) && BreakSkip != Pc)
        {
            BreakSkip = Pc;
            return Halt.Breakpoint;
        }
        return Halt.None;
    }

    Halt ExecuteOne(string[] lines, bool trace)
    {
        int line = Pc;
        string raw = lines[line - 1];
        var watch = Stopwatch.StartNew();
        try
        {
            if (trace)
                Messages.Add("第 " + line + " 行  " + raw.Trim());
            Execute(raw);
            watch.Stop();
            Note(line, watch.Elapsed.TotalMilliseconds);
            Steps++;
            BreakSkip = 0;
            Pc = line + 1;
            if (Pc > lines.Length)
            {
                Done = true;
                return Halt.Done;
            }
            return Halt.None;
        }
        catch (Exception ex)
        {
            Error = ex.Message;
            ErrorLine = line;
            Pc = line;
            return Halt.Error;
        }
    }

    static bool IsSkippable(string raw)
    {
        string text = raw.Trim();
        return text.Length == 0 || text.StartsWith('*') || text.StartsWith("//");
    }

    void Execute(string raw)
    {
        string text = raw.Trim();
        var assign = System.Text.RegularExpressions.Regex.Match(text, @"^([A-Za-z_]\w*)\s*:=\s*(.+)$");
        if (assign.Success)
        {
            Vars[assign.Groups[1].Value] = Eval(assign.Groups[2].Value.Trim());
            return;
        }
        ParsedCall? call = OperatorCall.ParseLine(text);
        if (call == null)
            throw new InvalidOperationException("看不懂這一行");
        Dispatch(call.Value.Name, call.Value.Args);
    }

    void Dispatch(string name, string[] args)
    {
        string bare = name.Contains('.') ? name.Split('.')[^1] : name;
        switch (bare.ToLowerInvariant())
        {
            case "read_image":
                Need(args, 2);
                var image = Operators.ReadImage(Resolve(Unquote(args[1]), ProjectFolder));
                Set(args[0], image);
                Image = image;
                Shapes = [];
                return;
            case "threshold":
                Need(args, 4);
                var region = Operators.Threshold(ImageOf(args[0]), IntOf(args[2]), IntOf(args[3]));
                Set(args[1], region);
                Show(region);
                return;
            case "connection":
                Need(args, 2);
                double distance = args.Length > 2 ? Num(args[2]) : 1;
                Region[] parts = Operators.Connection(RegionOf(args[0]), distance);
                Set(args[1], parts);
                Show(parts);
                return;
            case "count_obj":
                Need(args, 2);
                int count = Get(args[0]) is Region[] many ? Operators.Count(many) : Operators.Count(RegionOf(args[0]));
                Set(args[1], (double)count);
                return;
            case "area_center":
                Need(args, 4);
                Operators.AreaCenter[] centers = Get(args[0]) is Region[] list
                    ? Operators.AreaCenterOf(list)
                    : Operators.AreaCenterOf(RegionOf(args[0]));
                Set(args[1], centers.Select(item => (double)item.Area).ToArray());
                Set(args[2], centers.Select(item => item.Row).ToArray());
                Set(args[3], centers.Select(item => item.Column).ToArray());
                return;
            case "smallest_rectangle1":
                Need(args, 5);
                AxisRectangle[] boxes = Get(args[0]) is Region[] regions
                    ? Operators.Bounds(regions)
                    : Operators.Bounds(RegionOf(args[0]));
                Set(args[1], boxes.Select(item => (double)item.Row1).ToArray());
                Set(args[2], boxes.Select(item => (double)item.Column1).ToArray());
                Set(args[3], boxes.Select(item => (double)item.Row2).ToArray());
                Set(args[4], boxes.Select(item => (double)item.Column2).ToArray());
                return;
            case "intensity":
                Need(args, 4);
                Intensity[] stats = Get(args[0]) is Region[] areas
                    ? Operators.IntensityOf(areas, ImageOf(args[1]))
                    : Operators.IntensityOf(RegionOf(args[0]), ImageOf(args[1]));
                Set(args[2], stats.Select(item => item.Mean).ToArray());
                Set(args[3], stats.Select(item => item.Deviation).ToArray());
                return;
            case "mean_image":
                Need(args, 4);
                var mean = Operators.MeanImage(ImageOf(args[0]), IntOf(args[2]), IntOf(args[3]));
                Set(args[1], mean);
                Image = mean;
                return;
            case "gauss_image":
                Need(args, 3);
                var smooth = Operators.GaussImage(ImageOf(args[0]), IntOf(args[2]));
                Set(args[1], smooth);
                Image = smooth;
                return;
            case "reduce_domain":
                Need(args, 3);
                var reduced = Operators.ReduceDomain(ImageOf(args[0]), RegionOf(args[1]));
                Set(args[2], reduced);
                Image = reduced;
                return;
            case "fill_up":
                Need(args, 2);
                var filled = Operators.FillUp(RegionOf(args[0]));
                Set(args[1], filled);
                Show(filled);
                return;
            case "difference":
                Need(args, 3);
                var diff = Operators.Difference(RegionOf(args[0]), RegionOf(args[1]));
                Set(args[2], diff);
                Show(diff);
                return;
            case "union1":
                Need(args, 2);
                var union = Get(args[0]) is Region[] group ? Operators.Union(group) : RegionOf(args[0]);
                Set(args[1], union);
                Show(union);
                return;
            case "select_shape":
                Need(args, 6);
                string feature = Unquote(args[2]);
                var selected = Get(args[0]) is Region[] input
                    ? Operators.SelectShape(input, new[] { feature }, Unquote(args[3]), new[] { Num(args[4]) }, new[] { Num(args[5]) })
                    : Operators.SelectShape(RegionOf(args[0]), new[] { feature }, Unquote(args[3]), new[] { Num(args[4]) }, new[] { Num(args[5]) });
                Set(args[1], selected);
                Show(selected);
                return;
            case "edges_sub_pix":
                Need(args, 6);
                if (!Unquote(args[2]).Equals("canny", StringComparison.OrdinalIgnoreCase))
                    throw new InvalidOperationException("這版只做 canny");
                EdgePoint[][] edges = Operators.EdgesSubPix(ImageOf(args[0]), Num(args[3]), Num(args[4]), Num(args[5]));
                Set(args[1], edges);
                ShowEdges(edges);
                return;
            case "threshold_sub_pix":
                Need(args, 3);
                SubpixFigure figure = Operators.ThresholdSubPix(ImageOf(args[0]), Num(args[2]));
                Set(args[1], figure);
                Shapes = SubpixFigure.Overlays(figure, 2);
                return;
            case "dev_display":
                Need(args, 1);
                Get(args[0]);
                return;
            default:
                throw new InvalidOperationException("還沒接上這個算子：" + name);
        }
    }

    void Show(Region region) => Shapes = Operators.Overlays(region);

    void Show(Region[] regions) => Shapes = Operators.Overlays(regions);

    void ShowEdges(EdgePoint[][] edges)
    {
        var overlays = new List<Overlay>();
        foreach (EdgePoint[] chain in edges)
        {
            if (chain.Length < 2)
                continue;
            overlays.Add(Overlay.Polyline(chain.Select(point => point.Row).ToArray(), chain.Select(point => point.Column).ToArray(), 0xFF00E05A));
        }
        Shapes = overlays.ToArray();
    }

    object Get(string token)
    {
        string name = Ident(token);
        if (!Vars.TryGetValue(name, out object? value))
            throw new InvalidOperationException("還沒有變數 " + name);
        return value;
    }

    object Eval(string text)
    {
        if (IsName(text))
            return Get(text);
        if (text.Length >= 2 && text[0] is '\'' or '"')
            return Unquote(text);
        if (double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double number))
            return number;
        throw new InvalidOperationException("看不懂這個算式：" + text);
    }

    void Set(string token, object value)
    {
        string name = Ident(token);
        Vars[name] = value;
    }

    string Ident(string token)
    {
        string name = token.Trim();
        if (!IsName(name))
            throw new InvalidOperationException(name + " 必須是變數名稱");
        return name;
    }

    static bool IsName(string text) => System.Text.RegularExpressions.Regex.IsMatch(text, @"^[A-Za-z_]\w*$");

    ImageBuffer ImageOf(string token)
    {
        if (Eval(token) is ImageBuffer image)
            return image;
        throw new InvalidOperationException(token + " 必須是影像");
    }

    Region RegionOf(string token)
    {
        object value = Eval(token);
        if (value is Region region)
            return region;
        if (value is Region[] regions)
            return Operators.Union(regions);
        throw new InvalidOperationException(token + " 必須是區域");
    }

    double Num(string token)
    {
        object value = Eval(token);
        if (value is double number)
            return number;
        if (value is double[] many && many.Length == 1)
            return many[0];
        throw new InvalidOperationException(token + " 必須是數字");
    }

    int IntOf(string token) => (int)Math.Round(Num(token));

    static string Unquote(string token)
    {
        string text = token.Trim();
        if (text.Length >= 2 && ((text[0] == '\'' && text[^1] == '\'') || (text[0] == '"' && text[^1] == '"')))
            return text[1..^1];
        return text;
    }

    static void Need(string[] args, int count)
    {
        if (args.Length < count)
            throw new InvalidOperationException("參數不夠");
    }

    void Note(int line, double ms)
    {
        Times.TryGetValue(line, out var slot);
        Times[line] = (slot.Ms + ms, slot.Count + 1);
    }

    public static string Resolve(string path, string? projectFolder = null, string? repoRoot = null)
    {
        string text = path.Trim().Replace('\\', '/');
        if (text.Length == 0)
            return path;
        string native = text.Replace('/', Path.DirectorySeparatorChar);
        if (Path.IsPathRooted(native) && File.Exists(native))
            return native;
        if (!string.IsNullOrEmpty(projectFolder) && !Path.IsPathRooted(native))
        {
            string folder = Path.GetFullPath(projectFolder);
            string inside = Path.GetFullPath(Path.Combine(folder, native));
            if (IsInside(folder, inside) && File.Exists(inside))
                return inside;
        }
        string? repo = repoRoot ?? ProjectStore.FindRepoRoot();
        if (repo != null)
        {
            foreach (string candidate in new[]
            {
                Path.Combine(repo, native),
                Path.Combine(repo, "samples", native),
                Path.Combine(repo, "workspace", native),
            })
            {
                if (File.Exists(candidate))
                    return candidate;
            }
        }
        return path;
    }

    static bool IsInside(string folder, string candidate)
    {
        string root = Path.GetFullPath(folder).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
        string full = Path.GetFullPath(candidate);
        return full.StartsWith(root, StringComparison.OrdinalIgnoreCase);
    }
}

enum Halt { None, Yield, Breakpoint, Done, Error, Stopped }
