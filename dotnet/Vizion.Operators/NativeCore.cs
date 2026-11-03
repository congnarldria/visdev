using System.Reflection;
using System.Runtime.InteropServices;

namespace Vizion;

public static class NativeCore
{
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
        if (!IsLoaded)
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
}
