using System.IO;
using System.Text.RegularExpressions;

namespace Photonix;

sealed record ProjectProgram(string Project, string Name, string Path, string Source, IReadOnlyList<string> Images);

static class ProjectStore
{
    static readonly HashSet<string> ImageSuffix = new(StringComparer.OrdinalIgnoreCase)
    {
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp",
    };

    public static string? FindRepoRoot()
    {
        foreach (string start in new[] { AppContext.BaseDirectory, Directory.GetCurrentDirectory() })
        {
            var dir = new DirectoryInfo(start);
            while (dir != null)
            {
                if (Directory.Exists(Path.Combine(dir.FullName, "samples")) && Directory.Exists(Path.Combine(dir.FullName, "workspace")))
                    return dir.FullName;
                dir = dir.Parent;
            }
        }
        return null;
    }

    public static string ProjectsDir(string repo) => Path.Combine(repo, "workspace", "projects");

    public static string CleanProjectName(string? name)
    {
        string text = (name ?? "").Trim();
        if (text.Length == 0 || text is "." or ".." || text.IndexOfAny(['\\', '/', ':', '*', '?', '"', '<', '>', '|']) >= 0)
            throw new InvalidOperationException("專案名稱不正確");
        return text;
    }

    public static string CleanProgramName(string? name)
    {
        string text = Path.GetFileName((name ?? "").Trim());
        if (text.Length == 0)
            text = "program.viz";
        if (text is "." or ".." || text.IndexOfAny(['\\', '/', ':', '*', '?', '"', '<', '>', '|']) >= 0)
            throw new InvalidOperationException("程式名稱不正確");
        string lower = text.ToLowerInvariant();
        if (!lower.EndsWith(".hdv") && !lower.EndsWith(".viz"))
            text += ".viz";
        return text;
    }

    public static List<string> List(string repo)
    {
        string folder = ProjectsDir(repo);
        Directory.CreateDirectory(folder);
        return Directory.EnumerateDirectories(folder)
            .Where(path => File.Exists(Path.Combine(path, "visdev.json")))
            .Select(Path.GetFileName)
            .Where(name => !string.IsNullOrEmpty(name))
            .Cast<string>()
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToList();
    }

    public static string Create(string repo, string name)
    {
        string folder = Path.Combine(ProjectsDir(repo), CleanProjectName(name));
        if (Directory.Exists(folder))
            throw new InvalidOperationException("已經有這個專案");
        Directory.CreateDirectory(folder);
        File.WriteAllText(Path.Combine(folder, "visdev.json"), "{}\n");
        return Path.GetFileName(folder);
    }

    public static string Rename(string repo, string oldName, string newName)
    {
        string folder = Folder(repo, oldName);
        if (!Directory.Exists(folder) || !File.Exists(Path.Combine(folder, "visdev.json")))
            throw new InvalidOperationException("找不到專案");
        string targetName = CleanProjectName(newName);
        if (targetName == Path.GetFileName(folder))
            return targetName;
        string target = Path.Combine(ProjectsDir(repo), targetName);
        if (Directory.Exists(target))
            throw new InvalidOperationException("已經有這個專案");
        Directory.Move(folder, target);
        return targetName;
    }

    public static string Folder(string repo, string name)
    {
        string folder = Path.Combine(ProjectsDir(repo), CleanProjectName(name));
        if (!Directory.Exists(folder) || !File.Exists(Path.Combine(folder, "visdev.json")))
            throw new InvalidOperationException("找不到專案");
        return folder;
    }

    public static List<string> Programs(string repo, string name)
    {
        string folder = Folder(repo, name);
        return Directory.EnumerateFiles(folder)
            .Select(Path.GetFileName)
            .Where(file => file != null && (file.EndsWith(".viz", StringComparison.OrdinalIgnoreCase) || file.EndsWith(".hdv", StringComparison.OrdinalIgnoreCase)))
            .Cast<string>()
            .OrderBy(file => file, StringComparer.Ordinal)
            .ToList();
    }

    public static string Read(string repo, string project, string program)
    {
        string path = Path.Combine(Folder(repo, project), CleanProgramName(program));
        if (!File.Exists(path))
            throw new InvalidOperationException("找不到程式");
        return File.ReadAllText(path);
    }

    public static ProjectProgram Save(string repo, string project, string program, string source)
    {
        string folder = Folder(repo, project);
        string filename = CleanProgramName(program);
        string text = ProgramFile.Normalize(source);
        (text, List<string> images) = AdoptWorkspaceImages(repo, folder, text);
        string path = Path.Combine(folder, filename);
        File.WriteAllText(path, ProgramFile.ForDisk(text, path));
        return new ProjectProgram(Path.GetFileName(folder), filename, path, text.EndsWith('\n') ? text : text + "\n", images);
    }

    static (string Text, List<string> Images) AdoptWorkspaceImages(string repo, string folder, string source)
    {
        var copied = new List<string>();
        var assigned = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        string rewritten = Regex.Replace(source, @"(['""])([^'""\r\n]*)\1", match =>
        {
            string quote = match.Groups[1].Value;
            string body = match.Groups[2].Value;
            string? image = WorkspaceImage(repo, body);
            if (image == null)
                return match.Value;
            if (!assigned.TryGetValue(image, out string? name))
            {
                name = CopyImageInto(folder, image);
                assigned[image] = name;
                if (!copied.Contains(name) && !IsInside(folder, image))
                    copied.Add(name);
            }
            return quote + name + quote;
        });
        return (rewritten, copied);
    }

    static string? WorkspaceImage(string repo, string raw)
    {
        string text = raw.Trim().Replace('\\', '/');
        if (text.Length == 0 || text.StartsWith("samples/", StringComparison.OrdinalIgnoreCase))
            return null;
        string workspace = Path.GetFullPath(Path.Combine(repo, "workspace"));
        string candidate;
        if (Path.IsPathRooted(text))
            candidate = text;
        else if (text.StartsWith("workspace/", StringComparison.OrdinalIgnoreCase))
            candidate = Path.Combine(repo, text.Replace('/', Path.DirectorySeparatorChar));
        else
            return null;
        string resolved;
        try
        {
            resolved = Path.GetFullPath(candidate);
        }
        catch (Exception)
        {
            return null;
        }
        if (!IsInside(workspace, resolved) || !File.Exists(resolved))
            return null;
        if (!ImageSuffix.Contains(Path.GetExtension(resolved)))
            return null;
        return resolved;
    }

    static string CopyImageInto(string folder, string source)
    {
        if (IsInside(folder, source))
            return Path.GetFileName(source);
        string dest = Path.Combine(folder, Path.GetFileName(source));
        if (File.Exists(dest))
        {
            try
            {
                if (File.ReadAllBytes(dest).AsSpan().SequenceEqual(File.ReadAllBytes(source)))
                    return Path.GetFileName(dest);
            }
            catch (Exception)
            {
                // 檔名撞到就改一個新的。
            }
            int number = 2;
            string stem = Path.GetFileNameWithoutExtension(source);
            string suffix = Path.GetExtension(source);
            while (File.Exists(Path.Combine(folder, stem + "_" + number + suffix)))
                number++;
            dest = Path.Combine(folder, stem + "_" + number + suffix);
        }
        File.Copy(source, dest);
        return Path.GetFileName(dest);
    }

    static bool IsInside(string folder, string candidate)
    {
        try
        {
            string root = Path.GetFullPath(folder).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
            string full = Path.GetFullPath(candidate);
            return full.StartsWith(root, StringComparison.OrdinalIgnoreCase);
        }
        catch (Exception)
        {
            return false;
        }
    }
}
