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
            case "rgb1_to_gray":
                Need(args, 2);
                var gray = Operators.Rgb1ToGray(ImageOf(args[0]));
                Set(args[1], gray);
                Image = gray;
                return;
            case "invert_image":
                Need(args, 2);
                var inverted = Operators.InvertImage(ImageOf(args[0]));
                Set(args[1], inverted);
                Image = inverted;
                return;
            case "scale_image":
                Need(args, 4);
                var scaled = Operators.ScaleImage(ImageOf(args[0]), Num(args[2]), Num(args[3]));
                Set(args[1], scaled);
                Image = scaled;
                return;
            case "median_image":
                Need(args, 3);
                var median = Operators.MedianImage(ImageOf(args[0]), IntOf(args[2]));
                Set(args[1], median);
                Image = median;
                return;
            case "add_image":
                Need(args, 5);
                var added = Operators.AddImage(ImageOf(args[0]), ImageOf(args[1]), Num(args[3]), Num(args[4]));
                Set(args[2], added);
                Image = added;
                return;
            case "sub_image":
                Need(args, 5);
                var subtracted = Operators.SubImage(ImageOf(args[0]), ImageOf(args[1]), Num(args[3]), Num(args[4]));
                Set(args[2], subtracted);
                Image = subtracted;
                return;
            case "mult_image":
                Need(args, 5);
                var multiplied = Operators.MultImage(ImageOf(args[0]), ImageOf(args[1]), Num(args[3]), Num(args[4]));
                Set(args[2], multiplied);
                Image = multiplied;
                return;
            case "abs_diff_image":
                Need(args, 4);
                var difference = Operators.AbsDiffImage(ImageOf(args[0]), ImageOf(args[1]), Num(args[3]));
                Set(args[2], difference);
                Image = difference;
                return;
            case "min_image":
                Need(args, 3);
                var darker = Operators.MinImage(ImageOf(args[0]), ImageOf(args[1]));
                Set(args[2], darker);
                Image = darker;
                return;
            case "max_image":
                Need(args, 3);
                var brighter = Operators.MaxImage(ImageOf(args[0]), ImageOf(args[1]));
                Set(args[2], brighter);
                Image = brighter;
                return;
            case "div_image":
                Need(args, 5);
                var divided = Operators.DivImage(ImageOf(args[0]), ImageOf(args[1]), Num(args[3]), Num(args[4]));
                Set(args[2], divided);
                Image = divided;
                return;
            case "emphasize":
                Need(args, 5);
                var sharp = Operators.Emphasize(ImageOf(args[0]), IntOf(args[2]), IntOf(args[3]), Num(args[4]));
                Set(args[1], sharp);
                Image = sharp;
                return;
            case "equ_histo_image":
                Need(args, 2);
                var equalized = Operators.EquHistoImage(ImageOf(args[0]));
                Set(args[1], equalized);
                Image = equalized;
                return;
            case "scale_image_max":
                Need(args, 2);
                var stretched = Operators.ScaleImageMax(ImageOf(args[0]));
                Set(args[1], stretched);
                Image = stretched;
                return;
            case "illuminate":
                Need(args, 5);
                var flattened = Operators.Illuminate(ImageOf(args[0]), IntOf(args[2]), IntOf(args[3]), Num(args[4]));
                Set(args[1], flattened);
                Image = flattened;
                return;
            case "count_channels":
                Need(args, 2);
                Set(args[1], (double)Operators.CountChannels(ImageOf(args[0])));
                return;
            case "access_channel":
                Need(args, 3);
                var channel = Operators.AccessChannel(ImageOf(args[0]), IntOf(args[2]));
                Set(args[1], channel);
                Image = channel;
                return;
            case "compose3":
                Need(args, 4);
                var composed = Operators.Compose3(ImageOf(args[0]), ImageOf(args[1]), ImageOf(args[2]));
                Set(args[3], composed);
                Image = composed;
                return;
            case "decompose3":
                Need(args, 4);
                ImageBuffer[] planes = Operators.Decompose3(ImageOf(args[0]));
                Set(args[1], planes[0]);
                Set(args[2], planes[1]);
                Set(args[3], planes[2]);
                return;
            case "trans_from_rgb":
                Need(args, 7);
                ImageBuffer[] converted = Operators.TransFromRgb(ImageOf(args[0]), ImageOf(args[1]), ImageOf(args[2]), ColorName(args[6]));
                Set(args[3], converted[0]);
                Set(args[4], converted[1]);
                Set(args[5], converted[2]);
                return;
            case "pow_image":
                Need(args, 3);
                var powered = Operators.PowImage(ImageOf(args[0]), Num(args[2]));
                Set(args[1], powered);
                Image = powered;
                return;
            case "sqrt_image":
                Need(args, 2);
                var rooted = Operators.SqrtImage(ImageOf(args[0]));
                Set(args[1], rooted);
                Image = rooted;
                return;
            case "log_image":
                Need(args, 2);
                var logged = Operators.LogImage(ImageOf(args[0]), args.Length > 2 ? Word(args[2]) : "scaled");
                Set(args[1], logged);
                Image = logged;
                return;
            case "exp_image":
                Need(args, 2);
                var expanded = Operators.ExpImage(ImageOf(args[0]), args.Length > 2 ? Word(args[2]) : "scaled");
                Set(args[1], expanded);
                Image = expanded;
                return;
            case "gamma_image":
                Need(args, 3);
                var corrected = Operators.GammaImage(
                    ImageOf(args[0]),
                    Num(args[2]),
                    args.Length > 3 ? Num(args[3]) : 0,
                    args.Length > 4 ? Num(args[4]) : 0,
                    args.Length > 5 ? Num(args[5]) : 255,
                    args.Length <= 6 || EncodeFlag(args[6]));
                Set(args[1], corrected);
                Image = corrected;
                return;
            case "abs_image":
                Need(args, 2);
                var centered = Operators.AbsImage(ImageOf(args[0]));
                Set(args[1], centered);
                Image = centered;
                return;
            case "scale_image_range":
                Need(args, 4);
                var ranged = Operators.ScaleImageRange(ImageOf(args[0]), Num(args[2]), Num(args[3]));
                Set(args[1], ranged);
                Image = ranged;
                return;
            case "bit_and":
                Need(args, 3);
                var maskedAnd = Operators.BitAnd(ImageOf(args[0]), ImageOf(args[1]));
                Set(args[2], maskedAnd);
                Image = maskedAnd;
                return;
            case "bit_or":
                Need(args, 3);
                var maskedOr = Operators.BitOr(ImageOf(args[0]), ImageOf(args[1]));
                Set(args[2], maskedOr);
                Image = maskedOr;
                return;
            case "bit_xor":
                Need(args, 3);
                var maskedXor = Operators.BitXor(ImageOf(args[0]), ImageOf(args[1]));
                Set(args[2], maskedXor);
                Image = maskedXor;
                return;
            case "bit_not":
                Need(args, 2);
                var flipped = Operators.BitNot(ImageOf(args[0]));
                Set(args[1], flipped);
                Image = flipped;
                return;
            case "bit_mask":
                Need(args, 3);
                var masked = Operators.BitMask(ImageOf(args[0]), IntOf(args[2]));
                Set(args[1], masked);
                Image = masked;
                return;
            case "bit_slice":
                Need(args, 3);
                var sliced = Operators.BitSlice(ImageOf(args[0]), IntOf(args[2]));
                Set(args[1], sliced);
                Image = sliced;
                return;
            case "bit_lshift":
                Need(args, 2);
                var shiftedLeft = Operators.BitLShift(ImageOf(args[0]), args.Length > 2 ? IntOf(args[2]) : 1);
                Set(args[1], shiftedLeft);
                Image = shiftedLeft;
                return;
            case "bit_rshift":
                Need(args, 2);
                var shiftedRight = Operators.BitRShift(ImageOf(args[0]), args.Length > 2 ? IntOf(args[2]) : 1);
                Set(args[1], shiftedRight);
                Image = shiftedRight;
                return;
            case "highpass_image":
                Need(args, 4);
                var passed = Operators.HighpassImage(ImageOf(args[0]), IntOf(args[2]), IntOf(args[3]));
                Set(args[1], passed);
                Image = passed;
                return;
            case "deviation_image":
                Need(args, 4);
                var deviated = Operators.DeviationImage(ImageOf(args[0]), IntOf(args[2]), IntOf(args[3]));
                Set(args[1], deviated);
                Image = deviated;
                return;
            case "sigma_image":
                Need(args, 4);
                var sigma = Operators.SigmaImage(ImageOf(args[0]), IntOf(args[2]), IntOf(args[3]));
                Set(args[1], sigma);
                Image = sigma;
                return;
            case "gray_range_rect":
                Need(args, 4);
                var spanned = Operators.GrayRangeRect(ImageOf(args[0]), IntOf(args[2]), IntOf(args[3]));
                Set(args[1], spanned);
                Image = spanned;
                return;
            case "laplace":
                Need(args, 2);
                var laplacian = Operators.Laplace(
                    ImageOf(args[0]),
                    args.Length > 2 ? Word(args[2]) : "absolute",
                    args.Length > 3 ? IntOf(args[3]) : 3,
                    args.Length > 4 ? Word(args[4]) : "n_4");
                Set(args[1], laplacian);
                Image = laplacian;
                return;
            case "lut_trans":
                Need(args, 3);
                var looked = Operators.LutTrans(ImageOf(args[0]), LutOf(args[2]));
                Set(args[1], looked);
                Image = looked;
                return;
            case "rgb3_to_gray":
                Need(args, 4);
                var weighted = Operators.Rgb3ToGray(ImageOf(args[0]), ImageOf(args[1]), ImageOf(args[2]));
                Set(args[3], weighted);
                Image = weighted;
                return;
            case "gen_image_proto":
                Need(args, 3);
                var cleared = Operators.GenImageProto(ImageOf(args[0]), Num(args[2]));
                Set(args[1], cleared);
                Image = cleared;
                return;
            case "paint_gray":
                Need(args, 3);
                var painted = Operators.PaintGray(ImageOf(args[0]), ImageOf(args[1]));
                Set(args[2], painted);
                Image = painted;
                return;
            case "overpaint_gray":
                Need(args, 3);
                var covered = Operators.OverpaintGray(ImageOf(args[0]), ImageOf(args[1]));
                Set(args[2], covered);
                Image = covered;
                return;
            case "rank_rect":
                Need(args, 5);
                var ranked = Operators.RankRect(ImageOf(args[0]), IntOf(args[2]), IntOf(args[3]), IntOf(args[4]));
                Set(args[1], ranked);
                Image = ranked;
                return;
            case "median_rect":
                Need(args, 4);
                var medianRect = Operators.MedianRect(ImageOf(args[0]), IntOf(args[2]), IntOf(args[3]), args.Length > 4 ? Word(args[4]) : "continued");
                Set(args[1], medianRect);
                Image = medianRect;
                return;
            case "prewitt_amp":
                Need(args, 2);
                var prewitt = Operators.PrewittAmp(ImageOf(args[0]));
                Set(args[1], prewitt);
                Image = prewitt;
                return;
            case "roberts":
                Need(args, 2);
                var robertsEdge = Operators.Roberts(ImageOf(args[0]), args.Length > 2 ? Word(args[2]) : "sum_sqrt");
                Set(args[1], robertsEdge);
                Image = robertsEdge;
                return;
            case "binomial_filter":
                Need(args, 2);
                var binomial = Operators.BinomialFilter(ImageOf(args[0]), args.Length > 2 ? IntOf(args[2]) : 3, args.Length > 3 ? IntOf(args[3]) : 3);
                Set(args[1], binomial);
                Image = binomial;
                return;
            case "kirsch_amp":
                Need(args, 2);
                var kirsch = Operators.KirschAmp(ImageOf(args[0]));
                Set(args[1], kirsch);
                Image = kirsch;
                return;
            case "frei_amp":
                Need(args, 2);
                var frei = Operators.FreiAmp(ImageOf(args[0]));
                Set(args[1], frei);
                Image = frei;
                return;
            case "robinson_amp":
                Need(args, 2);
                var robinsonAmp = Operators.RobinsonAmp(ImageOf(args[0]));
                Set(args[1], robinsonAmp);
                Image = robinsonAmp;
                return;
            case "robinson_dir":
                Need(args, 2);
                var robinsonDir = Operators.RobinsonDir(ImageOf(args[0]));
                Set(args[1], robinsonDir);
                Image = robinsonDir;
                return;
            case "atan2_image":
                Need(args, 3);
                var aimed = Operators.Atan2Image(ImageOf(args[0]), ImageOf(args[1]));
                Set(args[2], aimed);
                Image = aimed;
                return;
            case "midrange_image":
                Need(args, 4);
                var midrange = Operators.MidrangeImage(ImageOf(args[0]), IntOf(args[2]), IntOf(args[3]));
                Set(args[1], midrange);
                Image = midrange;
                return;
            case "sin_image":
                Need(args, 2);
                var sined = Operators.SinImage(ImageOf(args[0]));
                Set(args[1], sined);
                Image = sined;
                return;
            case "cos_image":
                Need(args, 2);
                var cosined = Operators.CosImage(ImageOf(args[0]));
                Set(args[1], cosined);
                Image = cosined;
                return;
            case "tan_image":
                Need(args, 2);
                var taned = Operators.TanImage(ImageOf(args[0]));
                Set(args[1], taned);
                Image = taned;
                return;
            case "smooth_image":
                Need(args, 2);
                var smoothed = Operators.SmoothImage(ImageOf(args[0]), args.Length > 2 ? Word(args[2]) : "mean", args.Length > 3 ? IntOf(args[3]) : 3);
                Set(args[1], smoothed);
                Image = smoothed;
                return;
            case "gen_image_gray_ramp":
                Need(args, 8);
                var ramp = Operators.GrayRamp(Num(args[1]), Num(args[2]), Num(args[3]), Num(args[4]), Num(args[5]), IntOf(args[6]), IntOf(args[7]));
                Set(args[0], ramp);
                Image = ramp;
                return;
            case "tile_images":
                Need(args, 3);
                var tiled = Operators.TileImages(ImageOf(args[0]), ImageOf(args[1]), args.Length > 3 ? Word(args[3]) : "horizontal");
                Set(args[2], tiled);
                Image = tiled;
                return;
            case "convol_image":
                Need(args, 3);
                var convolved = ConvolOf(args);
                Set(args[1], convolved);
                Image = convolved;
                return;
            case "asin_image":
                Need(args, 2);
                var asined = Operators.AsinImage(ImageOf(args[0]));
                Set(args[1], asined);
                Image = asined;
                return;
            case "acos_image":
                Need(args, 2);
                var acosed = Operators.AcosImage(ImageOf(args[0]));
                Set(args[1], acosed);
                Image = acosed;
                return;
            case "atan_image":
                Need(args, 2);
                var ataned = Operators.AtanImage(ImageOf(args[0]));
                Set(args[1], ataned);
                Image = ataned;
                return;
            case "bandpass_image":
                Need(args, 4);
                var band = Operators.BandpassImage(ImageOf(args[0]), IntOf(args[2]), IntOf(args[3]));
                Set(args[1], band);
                Image = band;
                return;
            case "add_image_border":
                Need(args, 3);
                var framed = Operators.AddImageBorder(ImageOf(args[0]), IntOf(args[2]), args.Length > 3 ? IntOf(args[3]) : 0);
                Set(args[1], framed);
                Image = framed;
                return;
            case "gen_gauss_pyramid":
                Need(args, 2);
                var pyramid = Operators.GaussPyramid(ImageOf(args[0]), args.Length > 2 ? Word(args[2]) : "constant", args.Length > 3 ? Num(args[3]) : 0.5);
                Set(args[1], pyramid);
                Image = pyramid;
                return;
            case "entropy_image":
                Need(args, 4);
                var entropy = Operators.EntropyImage(ImageOf(args[0]), IntOf(args[2]), IntOf(args[3]));
                Set(args[1], entropy);
                Image = entropy;
                return;
            case "set_grayval":
                Need(args, 5);
                var edited = Operators.SetGrayval(ImageOf(args[0]), IntOf(args[2]), IntOf(args[3]), IntOf(args[4]));
                Set(args[1], edited);
                Image = edited;
                return;
            case "get_image_type":
                Need(args, 2);
                Set(args[1], Operators.ImageType(ImageOf(args[0])));
                return;
            case "channels_to_image":
                Need(args, 4);
                var bundled = Operators.ChannelsToImage(ImageOf(args[0]), ImageOf(args[1]), ImageOf(args[2]));
                Set(args[3], bundled);
                Image = bundled;
                return;
            case "expand_domain_gray":
                Need(args, 4);
                var widened = Operators.ExpandDomainGray(ImageOf(args[0]), IntOf(args[2]), Num(args[3]));
                Set(args[1], widened);
                Image = widened;
                return;
            case "equ_histo_image_rect":
                Need(args, 6);
                var rectEqu = Operators.EquHistoImageRect(ImageOf(args[0]), IntOf(args[2]), IntOf(args[3]), IntOf(args[4]), IntOf(args[5]));
                Set(args[1], rectEqu);
                Image = rectEqu;
                return;
            case "gray_inside":
                Need(args, 2);
                var inside = Operators.GrayInside(ImageOf(args[0]));
                Set(args[1], inside);
                Image = inside;
                return;
            case "rank_image":
                Need(args, 2);
                var diskRank = Operators.RankImage(ImageOf(args[0]), args.Length > 2 ? IntOf(args[2]) : 1, args.Length > 3 ? IntOf(args[3]) : 0);
                Set(args[1], diskRank);
                Image = diskRank;
                return;
            case "gen_image_surface_first_order":
                Need(args, 9);
                var surface = Operators.SurfaceFirst(Word(args[1]), Num(args[2]), Num(args[3]), Num(args[4]), Num(args[5]), Num(args[6]), IntOf(args[7]), IntOf(args[8]));
                Set(args[0], surface);
                Image = surface;
                return;
            case "gen_image_surface_second_order":
                Need(args, 12);
                var curved = Operators.SurfaceSecond(
                    Word(args[1]), Num(args[2]), Num(args[3]), Num(args[4]), Num(args[5]), Num(args[6]), Num(args[7]),
                    Num(args[8]), Num(args[9]), IntOf(args[10]), IntOf(args[11]));
                Set(args[0], curved);
                Image = curved;
                return;
            case "region_to_mean":
                Need(args, 3);
                var averaged = Operators.RegionToMean(ImageOf(args[0]), RegionsOf(args[1]));
                Set(args[2], averaged);
                Image = averaged;
                return;
            case "gray_dilation":
                Need(args, 3);
                var seMax = Operators.GrayDilation(ImageOf(args[0]), RegionOf(args[1]));
                Set(args[2], seMax);
                Image = seMax;
                return;
            case "gray_erosion":
                Need(args, 3);
                var seMin = Operators.GrayErosion(ImageOf(args[0]), RegionOf(args[1]));
                Set(args[2], seMin);
                Image = seMin;
                return;
            case "gray_opening":
                Need(args, 3);
                var seOpened = Operators.GrayOpening(ImageOf(args[0]), RegionOf(args[1]));
                Set(args[2], seOpened);
                Image = seOpened;
                return;
            case "gray_closing":
                Need(args, 3);
                var seClosed = Operators.GrayClosing(ImageOf(args[0]), RegionOf(args[1]));
                Set(args[2], seClosed);
                Image = seClosed;
                return;
            case "gray_dilation_shape":
                Need(args, 5);
                var shapeMax = Operators.GrayDilationShape(ImageOf(args[0]), IntOf(args[2]), IntOf(args[3]), Word(args[4]));
                Set(args[1], shapeMax);
                Image = shapeMax;
                return;
            case "gray_erosion_shape":
                Need(args, 5);
                var shapeMin = Operators.GrayErosionShape(ImageOf(args[0]), IntOf(args[2]), IntOf(args[3]), Word(args[4]));
                Set(args[1], shapeMin);
                Image = shapeMin;
                return;
            case "gray_opening_shape":
                Need(args, 5);
                var shapeOpened = Operators.GrayOpeningShape(ImageOf(args[0]), IntOf(args[2]), IntOf(args[3]), Word(args[4]));
                Set(args[1], shapeOpened);
                Image = shapeOpened;
                return;
            case "gray_closing_shape":
                Need(args, 5);
                var shapeClosed = Operators.GrayClosingShape(ImageOf(args[0]), IntOf(args[2]), IntOf(args[3]), Word(args[4]));
                Set(args[1], shapeClosed);
                Image = shapeClosed;
                return;
            case "mean_image_shape":
                Need(args, 3);
                var maskMean = Operators.MeanImageShape(ImageOf(args[0]), RegionOf(args[1]));
                Set(args[2], maskMean);
                Image = maskMean;
                return;
            case "select_grayvalues_from_channels":
                Need(args, 3);
                var channelPick = Operators.SelectGrayvalues(ImageOf(args[0]), ImageOf(args[1]));
                Set(args[2], channelPick);
                Image = channelPick;
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
                Operators.DevDisplay();
                return;
            case "dev_set_draw":
                Operators.DevSetDraw();
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

    double[] LutOf(string token)
    {
        if (Eval(token) is double[] table)
            return table;
        throw new InvalidOperationException("Lut 要 256 個數，對應灰階 0 到 255");
    }

    IReadOnlyList<Region> RegionsOf(string token)
    {
        object value = Eval(token);
        if (value is Region region)
            return new[] { region };
        if (value is Region[] many)
            return many;
        throw new InvalidOperationException(token + " 必須是區域");
    }

    ImageBuffer ConvolOf(string[] args)
    {
        string kernel = args[2].Trim();
        int width = args.Length > 3 ? IntOf(args[3]) : 0;
        string margin = args.Length > 4 ? Word(args[4]) : "continued";
        if (kernel.Length >= 2 && (kernel[0] == '\'' || kernel[0] == '"'))
            return Operators.ConvolImage(ImageOf(args[0]), Word(kernel), null, width, margin);
        if (IsName(kernel) && Vars.TryGetValue(kernel, out object? value) && value is double[] weights)
            return Operators.ConvolImage(ImageOf(args[0]), null, weights, width, margin);
        return Operators.ConvolImage(ImageOf(args[0]), Word(kernel), null, width, margin);
    }

    string Word(string token)
    {
        string text = token.Trim();
        if (text.Length >= 2 && (text[0] == '\'' || text[0] == '"'))
            return Unquote(text);
        if (IsName(text) && Vars.TryGetValue(text, out object? value) && value is string named)
            return named;
        return text;
    }

    bool EncodeFlag(string token)
    {
        string text = Word(token).Trim().ToLowerInvariant();
        if (text == "true")
            return true;
        if (text == "false")
            return false;
        throw new InvalidOperationException("Encode 只能是 true 或 false");
    }

    string ColorName(string token)
    {
        string text = token.Trim();
        if (IsName(text))
        {
            object value = Get(text);
            if (value is string named)
                return named;
            throw new InvalidOperationException(token + " 必須是色彩空間名稱");
        }
        return Unquote(text);
    }

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
