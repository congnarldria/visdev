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
