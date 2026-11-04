using System.Reflection;
using System.Runtime.InteropServices;

namespace Vizion;

public static class NativeCore
{
    public static bool UseNative { get; set; } = true;
    public static bool IsLoaded { get; private set; }
    public static int CpuFlags { get; private set; }
    public static string LoadDetail { get; private set; } = "";

    static IntPtr Handle;

    static NativeCore()
    {
        try
        {
            string? path = FindLibrary();
            if (path == null)
            {
                LoadDetail = "找不到 vizion_core";
                return;
            }
#if NETFRAMEWORK
            Handle = LoadLibraryW(path);
            if (Handle == IntPtr.Zero)
            {
                LoadDetail = "LoadLibrary 失敗 " + path;
                return;
            }
#else
            Handle = NativeLibrary.Load(path);
            NativeLibrary.SetDllImportResolver(typeof(NativeCore).Assembly, Resolve);
#endif
            IsLoaded = true;
            CpuFlags = vizion_cpu_flags();
            LoadDetail = path;
        }
        catch (Exception ex)
        {
            IsLoaded = false;
            LoadDetail = ex.Message;
        }
    }

    public static bool TryThreshold(byte[] gray, int width, int height, int minGray, int maxGray, out Region region)
    {
        region = new Region(width, height, Array.Empty<int>(), Array.Empty<int>(), Array.Empty<int>());
        if (!IsLoaded || !UseNative)
            return false;
        try
        {
            int code = vizion_threshold_runs(gray, width, height, minGray, maxGray, null, null, null, 0, out int count);
            if (code != 0)
                return false;
            if (count == 0)
                return true;
            var rows = new int[count];
            var columns = new int[count];
            var lengths = new int[count];
            code = vizion_threshold_runs(gray, width, height, minGray, maxGray, rows, columns, lengths, count, out int written);
            if (code != 0 || written != count)
                return false;
            region = new Region(width, height, rows, columns, lengths);
            return true;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    static string? FindLibrary()
    {
        string arch = Environment.Is64BitProcess ? "x64" : "x86";
        string[] names = { "vizion_core.dll", "libvizion_core.so" };
        string? dir = AppContext.BaseDirectory;
        for (int hop = 0; hop < 8 && !string.IsNullOrEmpty(dir); hop++)
        {
            foreach (string name in names)
            {
                string direct = Path.Combine(dir, arch, name);
                if (File.Exists(direct))
                    return direct;
                string bin = Path.Combine(dir, "bin", arch, name);
                if (File.Exists(bin))
                    return bin;
                string native = Path.Combine(dir, "native", "vizion_core", "bin", arch, name);
                if (File.Exists(native))
                    return native;
                string nested = Path.Combine(dir, "dotnet", "native", "vizion_core", "bin", arch, name);
                if (File.Exists(nested))
                    return nested;
            }
            dir = Path.GetDirectoryName(dir);
        }
        return null;
    }

#if NETFRAMEWORK
    [DllImport("kernel32", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern IntPtr LoadLibraryW(string path);
#else
    static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName == "vizion_core")
            return Handle;
        return IntPtr.Zero;
    }
#endif

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_cpu_flags();

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_threshold_runs(
        byte[] gray, int width, int height, int minGray, int maxGray,
        int[]? rows, int[]? columns, int[]? lengths, int capacity, out int written);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_connection8(
        int[] rows, int[] columns, int[] lengths, int count, int height,
        int[] outRows, int[] outColumns, int[] outLengths, int[] outOffsets, int[] outCounts);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_area_center(
        int[] rows, int[] columns, int[] lengths, int count,
        out int area, out double row, out double column);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_bounds(
        int[] rows, int[] columns, int[] lengths, int count,
        out int row1, out int column1, out int row2, out int column2);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_gray_rect(byte[] source, byte[] destination, int width, int height, int maskHeight, int maskWidth, int dilate);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_gray_ellipse(byte[] source, byte[] destination, int width, int height, int radius, int dilate);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_region_rect(
        int[] rows, int[] columns, int[] lengths, int count, int width, int height,
        int kernelWidth, int kernelHeight, int dilate,
        int[]? outRows, int[]? outColumns, int[]? outLengths, int capacity, out int written);

    public static bool TryMean(byte[] source, byte[] destination, int width, int height, int maskWidth, int maskHeight)
    {
        if (!IsLoaded || !UseNative)
            return false;
        try
        {
            return vizion_mean(source, destination, width, height, maskWidth, maskHeight) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryGauss(byte[] source, byte[] destination, int width, int height, int size)
    {
        if (!IsLoaded || !UseNative)
            return false;
        try
        {
            return vizion_gauss(source, destination, width, height, size) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryShapePose(
        float[] modelX, float[] modelY, float[] modelGx, float[] modelGy,
        float[] gx, float[] gy, float[] magnitude, int width, int height, int scale,
        ShapeHit hit, double minContrast, out ShapeHit posed)
    {
        posed = hit;
        if (!IsLoaded || !UseNative || modelX.Length == 0)
            return false;
        try
        {
            int code = vizion_shape_pose(modelX, modelY, modelGx, modelGy, modelX.Length, gx, gy, magnitude, width, height, scale, hit.Row, hit.Column, hit.Angle, hit.Score, minContrast, out double row, out double column, out double angle, out double score);
            if (code != 0)
                return false;
            posed = new ShapeHit(row, column, angle, score);
            return true;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryShapeScore(
        float[] gx, float[] gy, float[] mag, int width, int height, double column, double row,
        float[] mx, float[] my, float[] mgx, float[] mgy, double minContrast, out double score)
    {
        score = 0;
        if (!IsLoaded || !UseNative || mx.Length == 0)
            return false;
        try
        {
            score = vizion_shape_score(gx, gy, mag, width, height, column, row, mx, my, mgx, mgy, mx.Length, minContrast);
            return true;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryNcc(
        float[] gray, int width, int height, float[] x, float[] y, float[] value,
        double column, double row, bool absolute, out double score)
    {
        score = 0;
        if (!IsLoaded || !UseNative || x.Length == 0)
            return false;
        try
        {
            score = vizion_ncc_correlate(gray, width, height, x, y, value, x.Length, column, row, absolute ? 1 : 0);
            return true;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryRotateNearest(byte[] source, byte[] destination, int width, int height, double cosine, double sine)
    {
        if (!IsLoaded || !UseNative || width < 1 || height < 1 || source.Length < width * height || destination.Length < width * height)
            return false;
        try
        {
            return vizion_rotate_nearest(source, destination, width, height, cosine, sine) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryFitLine(double[] cols, double[] rows, out double rowBegin, out double colBegin, out double rowEnd, out double colEnd)
    {
        rowBegin = 0;
        colBegin = 0;
        rowEnd = 0;
        colEnd = 0;
        if (!IsLoaded || !UseNative || cols.Length < 2)
            return false;
        try
        {
            return vizion_fit_line(cols, rows, cols.Length, out rowBegin, out colBegin, out rowEnd, out colEnd) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryFitCircle(double[] cols, double[] rows, out double row, out double col, out double radius)
    {
        row = 0;
        col = 0;
        radius = 0;
        if (!IsLoaded || !UseNative || cols.Length < 3)
            return false;
        try
        {
            int code = vizion_fit_circle(cols, rows, cols.Length, out row, out col, out radius);
            if (code == -2)
                throw new InvalidOperationException("這些點擬合不出圓");
            return code == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryEdgeChains(
        byte[] center, float[] magnitude, int width, int height, double low, double high,
        float[] shift, float[] normalX, float[] normalY, out EdgePoint[][] chains)
    {
        chains = Array.Empty<EdgePoint[]>();
        if (!IsLoaded || !UseNative || width < 1 || height < 1)
            return false;
        try
        {
            int code = vizion_edge_chains(center, magnitude, low, high, width, height, shift, normalX, normalY, null, null, null, null, 0, 0, out int pointCount, out int chainCount);
            if (code != 0)
                return false;
            if (chainCount == 0)
                return true;
            var rows = new double[pointCount];
            var columns = new double[pointCount];
            var offsets = new int[chainCount];
            var counts = new int[chainCount];
            code = vizion_edge_chains(center, magnitude, low, high, width, height, shift, normalX, normalY, rows, columns, offsets, counts, pointCount, chainCount, out int writtenPoints, out int writtenChains);
            if (code != 0 || writtenChains != chainCount || writtenPoints != pointCount)
                return false;
            chains = new EdgePoint[chainCount][];
            for (int chain = 0; chain < chainCount; chain++)
            {
                var line = new EdgePoint[counts[chain]];
                for (int point = 0; point < counts[chain]; point++)
                {
                    int index = offsets[chain] + point;
                    line[point] = new EdgePoint(rows[index], columns[index]);
                }
                chains[chain] = line;
            }
            return true;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryIntensity(byte[] gray, int width, int[] rows, int[] columns, int[] lengths, out double mean, out double deviation)
    {
        mean = 0;
        deviation = 0;
        if (!IsLoaded || !UseNative || width < 1)
            return false;
        try
        {
            return vizion_intensity(gray, width, rows, columns, lengths, rows.Length, out mean, out deviation) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryBgrGray(byte[] bgr, int count, byte[] gray)
    {
        if (!IsLoaded || !UseNative || count < 1 || bgr.Length < count * 3 || gray.Length < count)
            return false;
        try
        {
            return vizion_bgr_gray(bgr, count, gray) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryGrayArith(byte[] left, byte[]? right, byte[] destination, int count, int op, double mult, double add)
    {
        if (!IsLoaded || !UseNative || count < 1 || left.Length < count || destination.Length < count)
            return false;
        if (op != 0 && op != 1 && (right == null || right.Length < count))
            return false;
        try
        {
            return vizion_gray_arith(left, right, destination, count, op, mult, add) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryEqualize(byte[] source, byte[] destination, int count)
    {
        if (!IsLoaded || !UseNative || count < 1 || source.Length < count || destination.Length < count)
            return false;
        try
        {
            return vizion_equalize(source, destination, count) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryEqualizeBgr(byte[] source, byte[] destination, int count)
    {
        if (!IsLoaded || !UseNative || count < 1 || source.Length < count * 3 || destination.Length < count * 3)
            return false;
        try
        {
            return vizion_equalize_bgr(source, destination, count) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryScaleMax(byte[] source, byte[] destination, int count, byte[]? domain)
    {
        if (!IsLoaded || !UseNative || count < 1 || source.Length < count || destination.Length < count)
            return false;
        if (domain != null && domain.Length < count)
            return false;
        try
        {
            return vizion_scale_max(source, destination, count, domain) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TrySplitBgr(byte[] bgr, byte[] blue, byte[] green, byte[] red, int count)
    {
        if (!IsLoaded || !UseNative || count < 1 || bgr.Length < count * 3)
            return false;
        if (blue.Length < count || green.Length < count || red.Length < count)
            return false;
        try
        {
            return vizion_split_bgr(bgr, blue, green, red, count) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryMergeBgr(byte[] blue, byte[] green, byte[] red, byte[] bgr, int count)
    {
        if (!IsLoaded || !UseNative || count < 1 || bgr.Length < count * 3)
            return false;
        if (blue.Length < count || green.Length < count || red.Length < count)
            return false;
        try
        {
            return vizion_merge_bgr(blue, green, red, bgr, count) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryRgbConvert(byte[] red, byte[] green, byte[] blue, byte[] first, byte[] second, byte[] third, int count, int space)
    {
        if (!IsLoaded || !UseNative || count < 1 || space < 0 || space > 2)
            return false;
        if (red.Length < count || green.Length < count || blue.Length < count)
            return false;
        if (first.Length < count || second.Length < count || third.Length < count)
            return false;
        try
        {
            return vizion_rgb_convert(red, green, blue, first, second, third, count, space) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryIlluminate(byte[] source, byte[] destination, int width, int height, int maskWidth, int maskHeight, double factor, byte[]? domain)
    {
        if (!IsLoaded || !UseNative || width < 1 || height < 1 || maskWidth < 1 || maskHeight < 1)
            return false;
        int count = width * height;
        if (source.Length < count || destination.Length < count)
            return false;
        if (domain != null && domain.Length < count)
            return false;
        try
        {
            return vizion_illuminate(source, destination, width, height, maskWidth, maskHeight, factor, domain) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryGraySqrt(byte[] source, byte[] destination, int count) =>
        Ready(source, destination, count) && Call(() => vizion_gray_sqrt(source, destination, count));

    public static bool TryGrayPow(byte[] source, byte[] destination, int count, double exponent) =>
        Ready(source, destination, count) && exponent > 0 && Call(() => vizion_gray_pow(source, destination, count, exponent));

    public static bool TryGrayLog(byte[] source, byte[] destination, int count, bool natural) =>
        Ready(source, destination, count) && Call(() => vizion_gray_log(source, destination, count, natural ? 1 : 0));

    public static bool TryGrayExp(byte[] source, byte[] destination, int count, bool natural) =>
        Ready(source, destination, count) && Call(() => vizion_gray_exp(source, destination, count, natural ? 1 : 0));

    public static bool TryGrayAbs(byte[] source, byte[] destination, int count) =>
        Ready(source, destination, count) && Call(() => vizion_gray_abs(source, destination, count));

    public static bool TryGrayGamma(byte[] source, byte[] destination, int count, double gamma, double offset, double threshold, double maxGray, bool encode) =>
        Ready(source, destination, count) && Call(() => vizion_gray_gamma(source, destination, count, gamma, offset, threshold, maxGray, encode ? 1 : 0));

    public static bool TryGrayRange(byte[] source, byte[] destination, int count, double low, double high) =>
        Ready(source, destination, count) && Call(() => vizion_gray_range(source, destination, count, low, high));

    public static bool TryGrayBits(byte[] left, byte[]? right, byte[] destination, int count, int op)
    {
        if (!Ready(left, destination, count) || op < 0 || op > 3)
            return false;
        if (op != 3 && (right == null || right.Length < count))
            return false;
        return Call(() => vizion_gray_bits(left, right, destination, count, op));
    }

    static bool Ready(byte[] source, byte[] destination, int count) =>
        IsLoaded && UseNative && count >= 1 && source.Length >= count && destination.Length >= count;

    static bool Call(Func<int> invoke)
    {
        try
        {
            return invoke() == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryMedian(byte[] source, byte[] destination, int width, int height, int ksize)
    {
        if (!IsLoaded || !UseNative || width < 1 || height < 1 || ksize < 1 || (ksize & 1) == 0)
            return false;
        if (source.Length < width * height || destination.Length < width * height)
            return false;
        try
        {
            return vizion_median_image(source, destination, width, height, ksize) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryPyrDown(byte[] source, int width, int height, byte[] destination)
    {
        int destWidth = (width + 1) / 2;
        int destHeight = (height + 1) / 2;
        if (!IsLoaded || !UseNative || width < 1 || height < 1 || source.Length < width * height || destination.Length < destWidth * destHeight)
            return false;
        try
        {
            return vizion_pyrdown(source, width, height, destination) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TrySobel(byte[] gray, int width, int height, float[] gx, float[] gy, float[] magnitude)
    {
        if (!IsLoaded || !UseNative || gray.Length < width * height)
            return false;
        try
        {
            return vizion_sobel(gray, width, height, gx, gy, magnitude) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TrySubpix(
        float[] gray, int width, int height, float[] smooth, float[] deriv,
        float[] dx, float[] dy, float[] magnitude, float[] normalX, float[] normalY,
        float[] shift, byte[] center)
    {
        if (!IsLoaded || !UseNative)
            return false;
        try
        {
            return vizion_subpix(gray, width, height, smooth, smooth.Length, deriv, deriv.Length, dx, dy, magnitude, normalX, normalY, shift, center) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryRemap(byte[] source, int width, int height, double[] mapX, double[] mapY, byte[] destination)
    {
        if (!IsLoaded || !UseNative)
            return false;
        try
        {
            return vizion_remap(source, width, height, mapX, mapY, destination) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryMeasureRow(byte[] row, out double left, out double right, out bool found)
    {
        left = 0;
        right = 0;
        found = false;
        if (!IsLoaded || !UseNative)
            return false;
        try
        {
            int code = vizion_measure_row(row, row.Length, out left, out right);
            if (code < 0)
                return false;
            found = code == 0;
            return true;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryMeasureScan(
        byte[] gray, int width, int height, double column1, double row1, double column2, double row2,
        double sigma, double threshold, int transition, out MeasureEdge[] edges)
    {
        edges = Array.Empty<MeasureEdge>();
        if (!IsLoaded || !UseNative)
            return false;
        try
        {
            int code = vizion_measure_scan(gray, width, height, column1, row1, column2, row2, sigma, threshold, transition, null, null, null, null, 0, out int count);
            if (code != 0)
                return false;
            if (count == 0)
                return true;
            var rows = new double[count];
            var columns = new double[count];
            var amps = new double[count];
            var dists = new double[count];
            code = vizion_measure_scan(gray, width, height, column1, row1, column2, row2, sigma, threshold, transition, rows, columns, amps, dists, count, out int written);
            if (code != 0 || written != count)
                return false;
            edges = new MeasureEdge[count];
            for (int i = 0; i < count; i++)
                edges[i] = new MeasureEdge(rows[i], columns[i], amps[i], dists[i]);
            return true;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_fit_line(
        double[] cols, double[] rows, int count,
        out double rowBegin, out double colBegin, out double rowEnd, out double colEnd);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_fit_circle(
        double[] cols, double[] rows, int count, out double row, out double col, out double radius);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_edge_chains(
        byte[] center, float[] magnitude, double low, double high, int width, int height,
        float[] shift, float[] normalX, float[] normalY,
        double[]? rows, double[]? columns, int[]? offsets, int[]? counts,
        int pointCapacity, int chainCapacity, out int pointCount, out int chainCount);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_rotate_nearest(byte[] source, byte[] destination, int width, int height, double cosine, double sine);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_intensity(
        byte[] gray, int width, int[] rows, int[] columns, int[] lengths, int runs, out double mean, out double deviation);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_bgr_gray(byte[] bgr, int count, byte[] gray);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_gray_arith(byte[] left, byte[]? right, byte[] destination, int count, int op, double mult, double add);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_median_image(byte[] source, byte[] destination, int width, int height, int ksize);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_equalize(byte[] source, byte[] destination, int count);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_equalize_bgr(byte[] source, byte[] destination, int count);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_scale_max(byte[] source, byte[] destination, int count, byte[]? domain);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_split_bgr(byte[] bgr, byte[] blue, byte[] green, byte[] red, int count);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_merge_bgr(byte[] blue, byte[] green, byte[] red, byte[] bgr, int count);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_rgb_convert(
        byte[] red, byte[] green, byte[] blue, byte[] first, byte[] second, byte[] third, int count, int space);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_illuminate(
        byte[] source, byte[] destination, int width, int height,
        int maskWidth, int maskHeight, double factor, byte[]? domain);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_gray_sqrt(byte[] source, byte[] destination, int count);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_gray_pow(byte[] source, byte[] destination, int count, double exponent);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_gray_log(byte[] source, byte[] destination, int count, int natural);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_gray_exp(byte[] source, byte[] destination, int count, int natural);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_gray_abs(byte[] source, byte[] destination, int count);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_gray_gamma(
        byte[] source, byte[] destination, int count,
        double gamma, double offset, double threshold, double maxGray, int encode);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_gray_range(byte[] source, byte[] destination, int count, double low, double high);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_gray_bits(byte[] left, byte[]? right, byte[] destination, int count, int op);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_pyrdown(byte[] source, int width, int height, byte[] destination);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_sobel(byte[] gray, int width, int height, float[] gx, float[] gy, float[] magnitude);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_subpix(
        float[] gray, int width, int height, float[] smooth, int smoothLength, float[] deriv, int derivLength,
        float[] dx, float[] dy, float[] magnitude, float[] normalX, float[] normalY, float[] shift, byte[] center);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_remap(byte[] source, int width, int height, double[] mapX, double[] mapY, byte[] destination);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_measure_row(byte[] row, int width, out double left, out double right);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_measure_scan(
        byte[] gray, int width, int height, double column1, double row1, double column2, double row2,
        double sigma, double threshold, int transition,
        double[]? rows, double[]? columns, double[]? amplitudes, double[]? distances,
        int capacity, out int written);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_shape_pose(
        float[] modelX, float[] modelY, float[] modelGx, float[] modelGy, int modelCount,
        float[] gx, float[] gy, float[] magnitude, int width, int height, int scale,
        double row, double column, double angle, double score, double minContrast,
        out double outRow, out double outColumn, out double outAngle, out double outScore);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern double vizion_shape_score(
        float[] gx, float[] gy, float[] mag, int width, int height, double column, double row,
        float[] mx, float[] my, float[] mgx, float[] mgy, int count, double minContrast);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern double vizion_ncc_correlate(
        float[] gray, int width, int height, float[] x, float[] y, float[] value,
        int count, double column, double row, int absolute);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_mean(byte[] source, byte[] destination, int width, int height, int maskWidth, int maskHeight);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_gauss(byte[] source, byte[] destination, int width, int height, int size);

    [DllImport("vizion_core", CallingConvention = CallingConvention.Cdecl)]
    static extern int vizion_region_disk(
        int[] rows, int[] columns, int[] lengths, int count, int width, int height, double radius, int dilate,
        int[]? outRows, int[]? outColumns, int[]? outLengths, int capacity, out int written);

    public static bool TryConnection(Region region, out Region[] objects)
    {
        objects = Array.Empty<Region>();
        if (!IsLoaded || !UseNative)
            return false;
        int count = region.Rows.Length;
        if (count == 0)
            return true;
        try
        {
            var rows = new int[count];
            var columns = new int[count];
            var lengths = new int[count];
            var offsets = new int[count];
            var counts = new int[count];
            int found = vizion_connection8(region.Rows, region.Columns, region.Lengths, count, region.Height, rows, columns, lengths, offsets, counts);
            if (found < 0)
                return false;
            objects = new Region[found];
            for (int index = 0; index < found; index++)
            {
                int runCount = counts[index];
                int at = offsets[index];
                var objectRows = new int[runCount];
                var objectColumns = new int[runCount];
                var objectLengths = new int[runCount];
                Array.Copy(rows, at, objectRows, 0, runCount);
                Array.Copy(columns, at, objectColumns, 0, runCount);
                Array.Copy(lengths, at, objectLengths, 0, runCount);
                objects[index] = new Region(region.Width, region.Height, objectRows, objectColumns, objectLengths);
            }
            return true;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryAreaCenter(Region region, out int area, out double row, out double column)
    {
        area = 0;
        row = 0;
        column = 0;
        if (!IsLoaded || !UseNative)
            return false;
        try
        {
            return vizion_area_center(region.Rows, region.Columns, region.Lengths, region.Rows.Length, out area, out row, out column) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryGrayRect(byte[] source, int width, int height, int maskHeight, int maskWidth, bool dilate, byte[] destination)
    {
        if (!IsLoaded || !UseNative)
            return false;
        try
        {
            return vizion_gray_rect(source, destination, width, height, maskHeight, maskWidth, dilate ? 1 : 0) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryGrayEllipse(byte[] source, int width, int height, int radius, bool dilate, byte[] destination)
    {
        if (!IsLoaded || !UseNative)
            return false;
        try
        {
            return vizion_gray_ellipse(source, destination, width, height, radius, dilate ? 1 : 0) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryRegionRect(int[] rows, int[] columns, int[] lengths, int width, int height, int kernelWidth, int kernelHeight, bool dilate, out Region region)
    {
        return TryRegion(true, rows, columns, lengths, width, height, kernelWidth, kernelHeight, 0, dilate, out region);
    }

    public static bool TryRegionDisk(int[] rows, int[] columns, int[] lengths, int width, int height, double radius, bool dilate, out Region region)
    {
        return TryRegion(false, rows, columns, lengths, width, height, 0, 0, radius, dilate, out region);
    }

    static bool TryRegion(bool rectangle, int[] rows, int[] columns, int[] lengths, int width, int height, int kernelWidth, int kernelHeight, double radius, bool dilate, out Region region)
    {
        region = new Region(width, height, Array.Empty<int>(), Array.Empty<int>(), Array.Empty<int>());
        if (!IsLoaded || !UseNative)
            return false;
        try
        {
            int count;
            int code = rectangle
                ? vizion_region_rect(rows, columns, lengths, rows.Length, width, height, kernelWidth, kernelHeight, dilate ? 1 : 0, null, null, null, 0, out count)
                : vizion_region_disk(rows, columns, lengths, rows.Length, width, height, radius, dilate ? 1 : 0, null, null, null, 0, out count);
            if (code != 0)
                return false;
            if (count == 0)
                return true;
            var outRows = new int[count];
            var outColumns = new int[count];
            var outLengths = new int[count];
            int written;
            code = rectangle
                ? vizion_region_rect(rows, columns, lengths, rows.Length, width, height, kernelWidth, kernelHeight, dilate ? 1 : 0, outRows, outColumns, outLengths, count, out written)
                : vizion_region_disk(rows, columns, lengths, rows.Length, width, height, radius, dilate ? 1 : 0, outRows, outColumns, outLengths, count, out written);
            if (code != 0 || written != count)
                return false;
            region = new Region(width, height, outRows, outColumns, outLengths);
            return true;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    public static bool TryBounds(Region region, out int row1, out int column1, out int row2, out int column2)
    {
        row1 = 0;
        column1 = 0;
        row2 = 0;
        column2 = 0;
        if (!IsLoaded || !UseNative || region.Rows.Length == 0)
            return false;
        try
        {
            return vizion_bounds(region.Rows, region.Columns, region.Lengths, region.Rows.Length, out row1, out column1, out row2, out column2) == 0;
        }
        catch (DllNotFoundException)
        {
            IsLoaded = false;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }
}
