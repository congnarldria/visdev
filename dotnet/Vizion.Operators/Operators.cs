namespace Vizion;

/// <summary>把圖形視窗顯示出來。簽名和 WinForms VizionWindow.Show 相同，所以可以直接傳 window.Show。</summary>
public delegate void ViewShow(int width, int height, byte[] pixels, PixelLayout layout, Overlay[] overlays);

/// <summary>Vizion.read_image、Vizion.threshold、Vizion.connection。距離 1 是八連通。</summary>
public static partial class Operators
{
    /// <summary>和圖形視窗 16 色循環相同，0xAARRGGBB。</summary>
    public static readonly uint[] Colors =
    {
        0xFFFF3B30,
        0xFF00E05A,
        0xFF3D8BFF,
        0xFFFFE14A,
        0xFF2CE6E6,
        0xFFF04AD4,
        0xFFFF8A1E,
        0xFFC6FF4A,
        0xFF7AA2FF,
        0xFFFF6A9A,
        0xFF46F0C8,
        0xFFE6C14A,
        0xFFB388FF,
        0xFFFF7A3C,
        0xFF9BE7FF,
        0xFFFFFFFF,
    };

    public static ImageBuffer ReadImage(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("無法讀取影像");
        byte[] file;
        try
        {
            file = File.ReadAllBytes(path);
        }
        catch (Exception)
        {
            throw new IOException($"無法讀取影像 {path}");
        }
        if (file.Length >= 2 && file[0] == 0xFF && file[1] == 0xD8)
            throw new InvalidDataException("C# 讀圖目前支援 PNG，這個檔是 JPEG");
        try
        {
            return PngImage.Decode(file, path);
        }
        catch (InvalidDataException)
        {
            throw;
        }
        catch (Exception)
        {
            throw new InvalidDataException($"無法讀取影像 {path}");
        }
    }

    /// <summary>灰階落在兩端（含）才留下。彩色先轉灰階。最小值比最大值大時是空區域。</summary>
    public static Region Threshold(ImageBuffer image, int minGray, int maxGray)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (minGray > maxGray)
            return new Region(image.Width, image.Height, Array.Empty<int>(), Array.Empty<int>(), Array.Empty<int>());
        byte[] gray = image.GrayBytes();
        var rows = new List<int>();
        var columns = new List<int>();
        var lengths = new List<int>();
        for (int row = 0; row < image.Height; row++)
        {
            int offset = row * image.Width;
            int column = 0;
            while (column < image.Width)
            {
                if (gray[offset + column] < minGray || gray[offset + column] > maxGray)
                {
                    column++;
                    continue;
                }
                int start = column;
                column++;
                while (column < image.Width && gray[offset + column] >= minGray && gray[offset + column] <= maxGray)
                    column++;
                rows.Add(row);
                columns.Add(start);
                lengths.Add(column - start);
            }
        }
        var region = new Region(image.Width, image.Height, rows.ToArray(), columns.ToArray(), lengths.ToArray());
        if (image.Domain != null)
            return Intersection(region, image.Domain);
        return region;
    }

    /// <summary>八連通。距離小於 2 不合併斜角以外的空隙。物件依掃描遇到的第一個像素排序。</summary>
    public static Region[] Connection(Region region, double distance = 1)
    {
        if (region == null)
            throw new ArgumentNullException(nameof(region));
        if (distance < 0)
            throw new ArgumentOutOfRangeException(nameof(distance), "Distance 不能是負數");
        int count = region.Rows.Length;
        if (count == 0)
            return Array.Empty<Region>();
        var order = new int[count];
        for (int i = 0; i < count; i++)
            order[i] = i;
        Array.Sort(order, (left, right) =>
        {
            int byRow = region.Rows[left].CompareTo(region.Rows[right]);
            if (byRow != 0)
                return byRow;
            return region.Columns[left].CompareTo(region.Columns[right]);
        });
        var rowRuns = new List<int>[region.Height];
        for (int row = 0; row < region.Height; row++)
            rowRuns[row] = new List<int>();
        var runs = new (int Row, int Start, int End)[count];
        for (int index = 0; index < count; index++)
        {
            int source = order[index];
            int row = region.Rows[source];
            int start = region.Columns[source];
            int end = start + region.Lengths[source] - 1;
            runs[index] = (row, start, end);
            if ((uint)row < (uint)region.Height)
                rowRuns[row].Add(index);
        }
        var parent = new int[count];
        for (int i = 0; i < count; i++)
            parent[i] = i;

        int Find(int index)
        {
            while (parent[index] != index)
            {
                parent[index] = parent[parent[index]];
                index = parent[index];
            }
            return index;
        }

        void Union(int left, int right)
        {
            int rootLeft = Find(left);
            int rootRight = Find(right);
            if (rootLeft != rootRight)
                parent[rootRight] = rootLeft;
        }

        for (int row = 1; row < region.Height; row++)
        {
            var above = rowRuns[row - 1];
            int cursor = 0;
            foreach (int index in rowRuns[row])
            {
                int start = runs[index].Start;
                int end = runs[index].End;
                int low = start - 1;
                int high = end + 1;
                while (cursor < above.Count && runs[above[cursor]].End < low)
                    cursor++;
                int probe = cursor;
                while (probe < above.Count && runs[above[probe]].Start <= high)
                {
                    Union(index, above[probe]);
                    probe++;
                }
            }
        }
        var groups = new Dictionary<int, List<int>>();
        for (int index = 0; index < count; index++)
        {
            int root = Find(index);
            if (!groups.TryGetValue(root, out var list))
            {
                list = new List<int>();
                groups.Add(root, list);
            }
            list.Add(index);
        }
        var ordered = new List<List<int>>(groups.Values);
        ordered.Sort((left, right) =>
        {
            var a = runs[left[0]];
            var b = runs[right[0]];
            int byRow = a.Row.CompareTo(b.Row);
            if (byRow != 0)
                return byRow;
            return a.Start.CompareTo(b.Start);
        });
        var objects = new Region[ordered.Count];
        for (int objectIndex = 0; objectIndex < ordered.Count; objectIndex++)
        {
            var group = ordered[objectIndex];
            var rows = new int[group.Count];
            var columns = new int[group.Count];
            var lengths = new int[group.Count];
            for (int i = 0; i < group.Count; i++)
            {
                var run = runs[group[i]];
                rows[i] = run.Row;
                columns[i] = run.Start;
                lengths[i] = run.End - run.Start + 1;
            }
            objects[objectIndex] = new Region(region.Width, region.Height, rows, columns, lengths);
        }
        if (distance < 2 || objects.Length < 2)
            return objects;
        return MergeWithin(objects, distance);
    }

    public readonly struct AreaCenter
    {
        public AreaCenter(int area, double row, double column)
        {
            Area = area;
            Row = row;
            Column = column;
        }

        public int Area { get; }
        public double Row { get; }
        public double Column { get; }
    }

    /// <summary>還沒拆開的區域是 0 或 1。拆開之後的個數是陣列長度。</summary>
    public static int Count(Region region)
    {
        if (region == null)
            throw new ArgumentNullException(nameof(region));
        return region.Rows.Length == 0 ? 0 : 1;
    }

    public static int Count(IReadOnlyList<Region> regions)
    {
        if (regions == null)
            throw new ArgumentNullException(nameof(regions));
        return regions.Count;
    }

    public static AreaCenter[] AreaCenterOf(Region region)
    {
        if (region == null)
            throw new ArgumentNullException(nameof(region));
        var one = CenterOf(region);
        return one.Area == 0 ? Array.Empty<AreaCenter>() : new[] { one };
    }

    public static AreaCenter[] AreaCenterOf(IReadOnlyList<Region> regions)
    {
        if (regions == null)
            throw new ArgumentNullException(nameof(regions));
        var found = new List<AreaCenter>(regions.Count);
        for (int i = 0; i < regions.Count; i++)
        {
            var center = CenterOf(regions[i]);
            if (center.Area != 0)
                found.Add(center);
        }
        return found.ToArray();
    }

    static AreaCenter CenterOf(Region region)
    {
        long area = 0;
        double rowMoment = 0;
        double columnMoment = 0;
        for (int i = 0; i < region.Lengths.Length; i++)
        {
            int length = region.Lengths[i];
            if (length <= 0)
                continue;
            int start = region.Columns[i];
            area += length;
            rowMoment += (double)region.Rows[i] * length;
            columnMoment += length * (2.0 * start + length - 1) / 2.0;
        }
        if (area == 0)
            return new AreaCenter(0, 0, 0);
        return new AreaCenter((int)area, rowMoment / area, columnMoment / area);
    }

    public static Overlay[] Overlays(Region region)
    {
        if (region == null)
            throw new ArgumentNullException(nameof(region));
        return new[] { region.ToOverlay(Colors[0]) };
    }

    public static Overlay[] Overlays(IReadOnlyList<Region> regions)
    {
        if (regions == null)
            throw new ArgumentNullException(nameof(regions));
        var overlays = new Overlay[regions.Count];
        for (int i = 0; i < regions.Count; i++)
            overlays[i] = regions[i].ToOverlay(Colors[i % Colors.Length]);
        return overlays;
    }
}
