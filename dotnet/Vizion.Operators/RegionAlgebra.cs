namespace Vizion;

public readonly struct AxisRectangle
{
    public AxisRectangle(int row1, int column1, int row2, int column2)
    {
        Row1 = row1;
        Column1 = column1;
        Row2 = row2;
        Column2 = column2;
    }

    public int Row1 { get; }
    public int Column1 { get; }
    public int Row2 { get; }
    public int Column2 { get; }
    public int Width => Column2 - Column1 + 1;
    public int Height => Row2 - Row1 + 1;
}

public readonly struct Intensity
{
    public Intensity(double mean, double deviation)
    {
        Mean = mean;
        Deviation = deviation;
    }

    public double Mean { get; }
    public double Deviation { get; }
}

public static partial class Operators
{
    public static Region Union(Region left, Region right) => Combine(left, right, CombineKind.Union);

    public static Region Union(IReadOnlyList<Region> regions)
    {
        if (regions == null)
            throw new ArgumentNullException(nameof(regions));
        if (regions.Count == 0)
            throw new ArgumentException("沒有區域可以合併");
        var runs = new List<Run>();
        for (int i = 0; i < regions.Count; i++)
            Append(runs, regions[i]);
        return FromRuns(regions[0].Width, regions[0].Height, runs);
    }

    public static Region Intersection(Region left, Region right) => Combine(left, right, CombineKind.Intersection);

    public static Region Intersection(Region left, IReadOnlyList<Region> right) => Intersection(left, Union(right));

    public static Region Intersection(IReadOnlyList<Region> left, Region right) => Intersection(Union(left), right);

    public static Region Intersection(IReadOnlyList<Region> left, IReadOnlyList<Region> right) => Intersection(Union(left), Union(right));

    public static Region Difference(Region left, Region right) => Combine(left, right, CombineKind.Difference);

    public static Region Difference(Region left, IReadOnlyList<Region> right) => Difference(left, Union(right));

    public static Region Difference(IReadOnlyList<Region> left, Region right) => Difference(Union(left), right);

    public static Region Difference(IReadOnlyList<Region> left, IReadOnlyList<Region> right) => Difference(Union(left), Union(right));

    public static Region[] Concat(Region left, Region right) => Concat(new[] { left }, new[] { right });

    public static Region[] Concat(Region left, IReadOnlyList<Region> right) => Concat(new[] { left }, right);

    public static Region[] Concat(IReadOnlyList<Region> left, Region right) => Concat(left, new[] { right });

    public static Region[] Concat(IReadOnlyList<Region> left, IReadOnlyList<Region> right)
    {
        if (left == null || right == null)
            throw new ArgumentNullException(nameof(left));
        if (left.Count == 0)
            return right as Region[] ?? Copy(right);
        if (right.Count == 0)
            return left as Region[] ?? Copy(left);
        if (left[0].Width != right[0].Width || left[0].Height != right[0].Height)
            throw new ArgumentException("兩個區域的尺寸不同");
        var all = new Region[left.Count + right.Count];
        for (int i = 0; i < left.Count; i++)
            all[i] = left[i];
        for (int i = 0; i < right.Count; i++)
            all[left.Count + i] = right[i];
        return all;
    }

    static Region[] Copy(IReadOnlyList<Region> regions)
    {
        var copy = new Region[regions.Count];
        for (int i = 0; i < regions.Count; i++)
            copy[i] = regions[i];
        return copy;
    }

    public static Region Select(Region region, int index)
    {
        int count = region == null ? 0 : (region.Rows.Length == 0 ? 0 : 1);
        if (region == null)
            throw new ArgumentNullException(nameof(region));
        if (index != 1 || count == 0)
            throw new IndexOutOfRangeException($"物件索引 {index} 超出範圍 1..{count}");
        return region;
    }

    public static Region Select(IReadOnlyList<Region> regions, int index)
    {
        if (regions == null)
            throw new ArgumentNullException(nameof(regions));
        if (index < 1 || index > regions.Count)
            throw new IndexOutOfRangeException($"物件索引 {index} 超出範圍 1..{regions.Count}");
        return regions[index - 1];
    }

    public static Region[] Select(IReadOnlyList<Region> regions, int[] indexes)
    {
        if (regions == null || indexes == null)
            throw new ArgumentNullException(nameof(regions));
        var picked = new Region[indexes.Length];
        for (int i = 0; i < indexes.Length; i++)
            picked[i] = Select(regions, indexes[i]);
        return picked;
    }

    public static Region[] SelectShape(Region region, string[] features, string operation, double[] minimum, double[] maximum) =>
        SelectShape(new[] { region }, features, operation, minimum, maximum);

    public static Region[] SelectShape(IReadOnlyList<Region> regions, string[] features, string operation, double[] minimum, double[] maximum)
    {
        if (regions == null || features == null || minimum == null || maximum == null)
            throw new ArgumentNullException(nameof(regions));
        string mode = (operation ?? "").Trim().Trim('\'').ToLowerInvariant();
        if (mode != "and" && mode != "or")
            throw new ArgumentException("Operation 只能是 and 或 or");
        if (features.Length == 0)
            throw new ArgumentException("Features 不能是空的");
        if (minimum.Length != features.Length || maximum.Length != features.Length)
            throw new ArgumentException("Min 和 Max 的長度要和特徵一樣");
        var kept = new List<Region>();
        for (int index = 0; index < regions.Count; index++)
        {
            bool match = mode == "and";
            for (int feature = 0; feature < features.Length; feature++)
            {
                double value = Feature(regions[index], features[feature]);
                bool inside = minimum[feature] <= value && value <= maximum[feature];
                if (mode == "and")
                    match &= inside;
                else
                    match |= inside;
            }
            if (match)
                kept.Add(regions[index]);
        }
        return kept.ToArray();
    }

    public static Region Rectangle(int row1, int column1, int row2, int column2) =>
        RectangleCore(row1, column1, row2, column2, 0, 0);

    public static Region Rectangle(ImageBuffer image, int row1, int column1, int row2, int column2)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        return RectangleCore(row1, column1, row2, column2, image.Width, image.Height);
    }

    public static AxisRectangle[] Bounds(Region region) =>
        region.Rows.Length == 0 ? Array.Empty<AxisRectangle>() : new[] { BoundsOf(region) };

    public static AxisRectangle[] Bounds(IReadOnlyList<Region> regions)
    {
        if (regions == null)
            throw new ArgumentNullException(nameof(regions));
        var found = new List<AxisRectangle>();
        for (int i = 0; i < regions.Count; i++)
        {
            if (regions[i].Rows.Length == 0)
                continue;
            found.Add(BoundsOf(regions[i]));
        }
        return found.ToArray();
    }

    public static int[] GrayAt(ImageBuffer image, int row, int column) => GrayAt(image, new[] { row }, new[] { column });

    public static int[] GrayAt(ImageBuffer image, int[] rows, int[] columns)
    {
        if (image == null || rows == null || columns == null)
            throw new ArgumentNullException(nameof(image));
        if (rows.Length != columns.Length || rows.Length == 0)
            throw new ArgumentException("Row 和 Column 的長度要相同，而且不能是空的");
        for (int i = 0; i < rows.Length; i++)
        {
            if (rows[i] < 0 || columns[i] < 0 || rows[i] >= image.Height || columns[i] >= image.Width)
                throw new ArgumentOutOfRangeException(nameof(rows), "座標超出影像");
        }
        if (image.Channels == 1)
        {
            var values = new int[rows.Length];
            for (int i = 0; i < rows.Length; i++)
                values[i] = image.Pixels[rows[i] * image.Width + columns[i]];
            return values;
        }
        var color = new int[rows.Length * 3];
        for (int i = 0; i < rows.Length; i++)
        {
            int pixel = (rows[i] * image.Width + columns[i]) * 3;
            color[i * 3] = image.Pixels[pixel + 2];
            color[i * 3 + 1] = image.Pixels[pixel + 1];
            color[i * 3 + 2] = image.Pixels[pixel];
        }
        return color;
    }

    public static int Contains(Region region, int row, int column) => Hit(region, row, column);

    public static int Contains(IReadOnlyList<Region> regions, int row, int column)
    {
        if (regions == null)
            throw new ArgumentNullException(nameof(regions));
        for (int i = 0; i < regions.Count; i++)
        {
            if (Hit(regions[i], row, column) == 1)
                return 1;
        }
        return 0;
    }

    public static Intensity[] IntensityOf(Region region, ImageBuffer image) => IntensityOf(new[] { region }, image);

    public static Intensity[] IntensityOf(IReadOnlyList<Region> regions, ImageBuffer image)
    {
        if (regions == null || image == null)
            throw new ArgumentNullException(nameof(regions));
        if (regions.Count > 0 && (regions[0].Width != image.Width || regions[0].Height != image.Height))
            throw new ArgumentException("影像和區域的尺寸不同");
        byte[] gray = image.GrayBytes();
        var found = new Intensity[regions.Count];
        for (int index = 0; index < regions.Count; index++)
        {
            long count = 0;
            double sum = 0;
            var region = regions[index];
            for (int i = 0; i < region.Rows.Length; i++)
            {
                int offset = region.Rows[i] * image.Width + region.Columns[i];
                for (int column = 0; column < region.Lengths[i]; column++)
                {
                    sum += gray[offset + column];
                    count++;
                }
            }
            if (count == 0)
            {
                found[index] = new Intensity(0, 0);
                continue;
            }
            double mean = sum / count;
            double square = 0;
            for (int i = 0; i < region.Rows.Length; i++)
            {
                int offset = region.Rows[i] * image.Width + region.Columns[i];
                for (int column = 0; column < region.Lengths[i]; column++)
                {
                    double delta = gray[offset + column] - mean;
                    square += delta * delta;
                }
            }
            found[index] = new Intensity(mean, Math.Sqrt(square / count));
        }
        return found;
    }

    public static Region FillUp(Region region) => Fill(region);

    public static Region FillUp(IReadOnlyList<Region> regions) => Fill(Union(regions));

    public static Region Complement(Region region) => Invert(region);

    public static Region Complement(IReadOnlyList<Region> regions) => Invert(Union(regions));

    public static Region DilateRectangle(Region region, int width, int height) =>
        Morphology(new[] { region }, width, height, "dilate");

    public static Region DilateRectangle(IReadOnlyList<Region> regions, int width, int height) =>
        Morphology(regions, width, height, "dilate");

    public static Region ErodeRectangle(Region region, int width, int height) =>
        Morphology(new[] { region }, width, height, "erode");

    public static Region ErodeRectangle(IReadOnlyList<Region> regions, int width, int height) =>
        Morphology(regions, width, height, "erode");

    public static Region OpenRectangle(Region region, int width, int height) =>
        Morphology(new[] { region }, width, height, "open");

    public static Region OpenRectangle(IReadOnlyList<Region> regions, int width, int height) =>
        Morphology(regions, width, height, "open");

    public static Region CloseRectangle(Region region, int width, int height) =>
        Morphology(new[] { region }, width, height, "close");

    public static Region CloseRectangle(IReadOnlyList<Region> regions, int width, int height) =>
        Morphology(regions, width, height, "close");

    static Region[] MergeWithin(Region[] objects, double distance)
    {
        int count = objects.Length;
        var boxes = new AxisRectangle[count];
        for (int i = 0; i < count; i++)
            boxes[i] = BoundsOf(objects[i]);
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
            if (rootLeft == rootRight)
                return;
            if (rootLeft < rootRight)
                parent[rootRight] = rootLeft;
            else
                parent[rootLeft] = rootRight;
        }
        var maps = new float[count][];
        for (int left = 0; left < count; left++)
        {
            for (int right = left + 1; right < count; right++)
            {
                if (Find(left) == Find(right))
                    continue;
                if (!BoxesWithin(boxes[left], boxes[right], distance))
                    continue;
                if (maps[left] == null)
                    maps[left] = Chamfer(objects[left]);
                if (Nearest(maps[left], objects[right]) <= distance + 1e-3)
                    Union(left, right);
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
        var keys = new List<int>(groups.Keys);
        keys.Sort();
        var merged = new Region[keys.Count];
        for (int index = 0; index < keys.Count; index++)
        {
            var runs = new List<Run>();
            foreach (int member in groups[keys[index]])
                Append(runs, objects[member]);
            merged[index] = FromRuns(objects[0].Width, objects[0].Height, runs);
        }
        return merged;
    }

    static bool BoxesWithin(AxisRectangle left, AxisRectangle right, double distance)
    {
        double gapColumn = 0;
        if (left.Column2 < right.Column1)
            gapColumn = right.Column1 - left.Column2;
        else if (right.Column2 < left.Column1)
            gapColumn = left.Column1 - right.Column2;
        double gapRow = 0;
        if (left.Row2 < right.Row1)
            gapRow = right.Row1 - left.Row2;
        else if (right.Row2 < left.Row1)
            gapRow = left.Row1 - right.Row2;
        return Math.Sqrt(gapColumn * gapColumn + gapRow * gapRow) <= distance;
    }

    static float Nearest(float[] distance, Region region)
    {
        float best = float.MaxValue;
        for (int i = 0; i < region.Rows.Length; i++)
        {
            int offset = region.Rows[i] * region.Width + region.Columns[i];
            for (int column = 0; column < region.Lengths[i]; column++)
            {
                float value = distance[offset + column];
                if (value < best)
                    best = value;
            }
        }
        return best;
    }

    /// <summary>和 cv2.distanceTransform(..., DIST_L2, 5) 相同的 5×5 權重。</summary>
    static float[] Chamfer(Region region)
    {
        int width = region.Width;
        int height = region.Height;
        float orthogonal = 1f;
        float diagonal = 1.4f;
        float knight = 2.1969f;
        float far = 1e6f;
        var distance = new float[width * height];
        for (int i = 0; i < distance.Length; i++)
            distance[i] = far;
        for (int i = 0; i < region.Rows.Length; i++)
        {
            int offset = region.Rows[i] * width + region.Columns[i];
            for (int column = 0; column < region.Lengths[i]; column++)
                distance[offset + column] = 0f;
        }
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
                distance[row * width + column] = Relax(distance, width, height, row, column, orthogonal, diagonal, knight, forward: true);
        }
        for (int row = height - 1; row >= 0; row--)
        {
            for (int column = width - 1; column >= 0; column--)
                distance[row * width + column] = Relax(distance, width, height, row, column, orthogonal, diagonal, knight, forward: false);
        }
        return distance;
    }

    static float Relax(float[] distance, int width, int height, int row, int column, float orthogonal, float diagonal, float knight, bool forward)
    {
        float value = distance[row * width + column];
        int direction = forward ? -1 : 1;
        value = Take(distance, width, height, row, column + direction, value, orthogonal);
        value = Take(distance, width, height, row, column + 2 * direction, value, orthogonal + orthogonal);
        value = Take(distance, width, height, row + direction, column, value, orthogonal);
        value = Take(distance, width, height, row + direction, column - 1, value, diagonal);
        value = Take(distance, width, height, row + direction, column + 1, value, diagonal);
        value = Take(distance, width, height, row + direction, column - 2, value, knight);
        value = Take(distance, width, height, row + direction, column + 2, value, knight);
        value = Take(distance, width, height, row + 2 * direction, column - 1, value, knight);
        value = Take(distance, width, height, row + 2 * direction, column + 1, value, knight);
        return value;
    }

    static float Take(float[] distance, int width, int height, int row, int column, float value, float step)
    {
        if ((uint)row >= (uint)height || (uint)column >= (uint)width)
            return value;
        float candidate = distance[row * width + column] + step;
        return candidate < value ? candidate : value;
    }

    enum CombineKind { Union, Intersection, Difference }

    static Region Combine(Region left, Region right, CombineKind kind)
    {
        if (left == null || right == null)
            throw new ArgumentNullException(nameof(left));
        if (left.Width != right.Width || left.Height != right.Height)
            throw new ArgumentException("兩個區域的尺寸不同");
        var leftRows = Intervals(left);
        var rightRows = Intervals(right);
        var runs = new List<Run>();
        var keys = new SortedSet<int>(leftRows.Keys);
        foreach (int key in rightRows.Keys)
            keys.Add(key);
        foreach (int row in keys)
        {
            leftRows.TryGetValue(row, out var leftRuns);
            rightRows.TryGetValue(row, out var rightRuns);
            leftRuns ??= new List<(int, int)>();
            rightRuns ??= new List<(int, int)>();
            IEnumerable<(int Start, int End)> produced = kind switch
            {
                CombineKind.Union => MergeIntervals(leftRuns, rightRuns),
                CombineKind.Intersection => IntersectIntervals(leftRuns, rightRuns),
                _ => SubtractIntervals(leftRuns, rightRuns),
            };
            foreach (var interval in produced)
                runs.Add(new Run(row, interval.Start, interval.End));
        }
        return FromRuns(left.Width, left.Height, runs);
    }

    static Dictionary<int, List<(int Start, int End)>> Intervals(Region region)
    {
        var rows = new Dictionary<int, List<(int Start, int End)>>();
        for (int i = 0; i < region.Rows.Length; i++)
        {
            int row = region.Rows[i];
            if (!rows.TryGetValue(row, out var list))
            {
                list = new List<(int, int)>();
                rows.Add(row, list);
            }
            list.Add((region.Columns[i], region.Columns[i] + region.Lengths[i] - 1));
        }
        foreach (var list in rows.Values)
            list.Sort((left, right) => left.Start.CompareTo(right.Start));
        return rows;
    }

    static List<(int Start, int End)> MergeIntervals(List<(int Start, int End)> left, List<(int Start, int End)> right)
    {
        var all = new List<(int Start, int End)>(left.Count + right.Count);
        all.AddRange(left);
        all.AddRange(right);
        all.Sort((a, b) => a.Start.CompareTo(b.Start));
        var merged = new List<(int Start, int End)>();
        foreach (var interval in all)
        {
            if (merged.Count == 0 || interval.Start > merged[merged.Count - 1].End + 1)
                merged.Add(interval);
            else
            {
                var last = merged[merged.Count - 1];
                merged[merged.Count - 1] = (last.Start, Math.Max(last.End, interval.End));
            }
        }
        return merged;
    }

    static List<(int Start, int End)> IntersectIntervals(List<(int Start, int End)> left, List<(int Start, int End)> right)
    {
        var found = new List<(int Start, int End)>();
        int index = 0;
        foreach (var interval in left)
        {
            while (index < right.Count && right[index].End < interval.Start)
                index++;
            int probe = index;
            while (probe < right.Count && right[probe].Start <= interval.End)
            {
                int start = Math.Max(interval.Start, right[probe].Start);
                int end = Math.Min(interval.End, right[probe].End);
                if (start <= end)
                    found.Add((start, end));
                probe++;
            }
        }
        return found;
    }

    static List<(int Start, int End)> SubtractIntervals(List<(int Start, int End)> left, List<(int Start, int End)> right)
    {
        var found = new List<(int Start, int End)>();
        foreach (var interval in left)
        {
            int cursor = interval.Start;
            foreach (var cut in right)
            {
                if (cut.End < cursor)
                    continue;
                if (cut.Start > interval.End)
                    break;
                if (cut.Start > cursor)
                    found.Add((cursor, Math.Min(interval.End, cut.Start - 1)));
                cursor = Math.Max(cursor, cut.End + 1);
                if (cursor > interval.End)
                    break;
            }
            if (cursor <= interval.End)
                found.Add((cursor, interval.End));
        }
        return found;
    }

    static Region Fill(Region region)
    {
        int width = region.Width;
        int height = region.Height;
        var foreground = new byte[width * height];
        for (int i = 0; i < region.Rows.Length; i++)
        {
            int offset = region.Rows[i] * width + region.Columns[i];
            for (int column = 0; column < region.Lengths[i]; column++)
                foreground[offset + column] = 1;
        }
        var outside = new byte[width * height];
        var stack = new Stack<int>();
        void Push(int row, int column)
        {
            if ((uint)row >= (uint)height || (uint)column >= (uint)width)
                return;
            int index = row * width + column;
            if (outside[index] != 0 || foreground[index] != 0)
                return;
            outside[index] = 1;
            stack.Push(index);
        }
        for (int column = 0; column < width; column++)
        {
            Push(0, column);
            Push(height - 1, column);
        }
        for (int row = 0; row < height; row++)
        {
            Push(row, 0);
            Push(row, width - 1);
        }
        while (stack.Count > 0)
        {
            int index = stack.Pop();
            int row = index / width;
            int column = index - row * width;
            Push(row - 1, column);
            Push(row + 1, column);
            Push(row, column - 1);
            Push(row, column + 1);
        }
        var runs = new List<Run>();
        for (int row = 0; row < height; row++)
        {
            int column = 0;
            while (column < width)
            {
                int index = row * width + column;
                if (foreground[index] == 0 && outside[index] != 0)
                {
                    column++;
                    continue;
                }
                int start = column;
                column++;
                while (column < width)
                {
                    int next = row * width + column;
                    if (foreground[next] == 0 && outside[next] != 0)
                        break;
                    column++;
                }
                runs.Add(new Run(row, start, column - 1));
            }
        }
        return FromRuns(width, height, runs);
    }

    static Region Invert(Region region)
    {
        var runs = new List<Run>();
        var rows = Intervals(region);
        for (int row = 0; row < region.Height; row++)
        {
            rows.TryGetValue(row, out var foreground);
            int cursor = 0;
            if (foreground != null)
            {
                foreach (var interval in foreground)
                {
                    if (interval.Start > cursor)
                        runs.Add(new Run(row, cursor, interval.Start - 1));
                    cursor = interval.End + 1;
                }
            }
            if (cursor < region.Width)
                runs.Add(new Run(row, cursor, region.Width - 1));
        }
        return FromRuns(region.Width, region.Height, runs);
    }

    static Region Morphology(IReadOnlyList<Region> regions, int width, int height, string mode)
    {
        if (regions == null || regions.Count == 0)
            throw new ArgumentException("沒有區域");
        int wide = OddSize(width, "Width");
        int high = OddSize(height, "Height");
        int radiusColumn = wide / 2;
        int radiusRow = high / 2;
        var source = RunsOf(Union(regions));
        List<Run> result = mode switch
        {
            "dilate" => Dilate(source, regions[0].Width, regions[0].Height, radiusColumn, radiusRow),
            "erode" => Erode(source, regions[0].Width, regions[0].Height, radiusColumn, radiusRow),
            "open" => Dilate(Erode(source, regions[0].Width, regions[0].Height, radiusColumn, radiusRow), regions[0].Width, regions[0].Height, radiusColumn, radiusRow),
            "close" => Erode(Dilate(source, regions[0].Width, regions[0].Height, radiusColumn, radiusRow), regions[0].Width, regions[0].Height, radiusColumn, radiusRow),
            _ => throw new ArgumentException(mode),
        };
        return FromRuns(regions[0].Width, regions[0].Height, result);
    }

    static int OddSize(int size, string name)
    {
        if (size < 1)
            throw new ArgumentOutOfRangeException(name, "矩形寬高至少要 1");
        return size % 2 == 0 ? size + 1 : size;
    }

    static List<Run> Dilate(List<Run> runs, int width, int height, int radiusColumn, int radiusRow)
    {
        var expanded = new List<Run>();
        foreach (var run in runs)
        {
            int start = Math.Max(0, run.Start - radiusColumn);
            int end = Math.Min(width - 1, run.End + radiusColumn);
            if (start > end)
                continue;
            int row1 = Math.Max(0, run.Row - radiusRow);
            int row2 = Math.Min(height - 1, run.Row + radiusRow);
            for (int row = row1; row <= row2; row++)
                expanded.Add(new Run(row, start, end));
        }
        return expanded;
    }

    static List<Run> Erode(List<Run> runs, int width, int height, int radiusColumn, int radiusRow)
    {
        var perRow = new List<(int Start, int End)>[height];
        for (int row = 0; row < height; row++)
            perRow[row] = new List<(int, int)>();
        foreach (var run in runs)
        {
            int start = run.Start + radiusColumn;
            int end = run.End - radiusColumn;
            if (start <= end && (uint)run.Row < (uint)height)
                perRow[run.Row].Add((start, end));
        }
        for (int row = 0; row < height; row++)
            perRow[row] = MergeIntervals(perRow[row], new List<(int, int)>());
        var result = new List<Run>();
        for (int row = radiusRow; row < height - radiusRow; row++)
        {
            var current = perRow[row - radiusRow];
            for (int other = row - radiusRow + 1; other <= row + radiusRow; other++)
                current = IntersectIntervals(current, perRow[other]);
            foreach (var interval in current)
                result.Add(new Run(row, interval.Start, interval.End));
        }
        return result;
    }

    static double Feature(Region region, string name)
    {
        string feature = (name ?? "").Trim().Trim('\'').Trim('"').ToLowerInvariant();
        if (feature == "center_row")
            feature = "row";
        if (feature == "center_column" || feature == "center_col")
            feature = "column";
        if (region.Rows.Length == 0)
        {
            if (feature is "area" or "row" or "column" or "width" or "height" or "ratio")
                return 0;
            throw new ArgumentException($"匯出的 select_shape 還沒有做特徵 {name}");
        }
        var box = BoundsOf(region);
        var center = CenterOf(region);
        return feature switch
        {
            "area" => region.Area,
            "row" => center.Row,
            "column" => center.Column,
            "width" => box.Width,
            "height" => box.Height,
            "ratio" => box.Height == 0 ? 0 : (double)box.Width / box.Height,
            _ => throw new ArgumentException($"匯出的 select_shape 還沒有做特徵 {name}"),
        };
    }

    static AxisRectangle BoundsOf(Region region)
    {
        int row1 = int.MaxValue;
        int column1 = int.MaxValue;
        int row2 = int.MinValue;
        int column2 = int.MinValue;
        for (int i = 0; i < region.Rows.Length; i++)
        {
            row1 = Math.Min(row1, region.Rows[i]);
            row2 = Math.Max(row2, region.Rows[i]);
            column1 = Math.Min(column1, region.Columns[i]);
            column2 = Math.Max(column2, region.Columns[i] + region.Lengths[i] - 1);
        }
        return new AxisRectangle(row1, column1, row2, column2);
    }

    static int Hit(Region region, int row, int column)
    {
        if (region == null)
            throw new ArgumentNullException(nameof(region));
        if (row < 0 || column < 0 || row >= region.Height || column >= region.Width)
            return 0;
        for (int i = 0; i < region.Rows.Length; i++)
        {
            if (region.Rows[i] != row)
                continue;
            int start = region.Columns[i];
            if (column >= start && column < start + region.Lengths[i])
                return 1;
        }
        return 0;
    }

    static Region RectangleCore(int row1, int column1, int row2, int column2, int imageWidth, int imageHeight)
    {
        if (row2 < row1 || column2 < column1)
            throw new ArgumentException("矩形座標不正確");
        int width = imageWidth;
        int height = imageHeight;
        if (width <= 0 || height <= 0)
        {
            width = column2 + 1;
            height = row2 + 1;
        }
        int top = Math.Max(0, Math.Min(height - 1, row1));
        int left = Math.Max(0, Math.Min(width - 1, column1));
        int bottom = Math.Max(0, Math.Min(height - 1, row2));
        int right = Math.Max(0, Math.Min(width - 1, column2));
        var runs = new List<Run>();
        for (int row = top; row <= bottom; row++)
            runs.Add(new Run(row, left, right));
        return FromRuns(width, height, runs);
    }

    readonly struct Run
    {
        public Run(int row, int start, int end)
        {
            Row = row;
            Start = start;
            End = end;
        }

        public int Row { get; }
        public int Start { get; }
        public int End { get; }
    }

    static List<Run> RunsOf(Region region)
    {
        var runs = new List<Run>(region.Rows.Length);
        Append(runs, region);
        return runs;
    }

    static void Append(List<Run> runs, Region region)
    {
        for (int i = 0; i < region.Rows.Length; i++)
            runs.Add(new Run(region.Rows[i], region.Columns[i], region.Columns[i] + region.Lengths[i] - 1));
    }

    static Region FromRuns(int width, int height, List<Run> runs)
    {
        runs.Sort((left, right) =>
        {
            int byRow = left.Row.CompareTo(right.Row);
            return byRow != 0 ? byRow : left.Start.CompareTo(right.Start);
        });
        var rows = new List<int>();
        var columns = new List<int>();
        var lengths = new List<int>();
        foreach (var run in runs)
        {
            if (run.End < run.Start || run.Row < 0 || run.Row >= height)
                continue;
            int start = Math.Max(0, run.Start);
            int end = Math.Min(width - 1, run.End);
            if (start > end)
                continue;
            if (rows.Count > 0 && rows[rows.Count - 1] == run.Row && start <= columns[columns.Count - 1] + lengths[lengths.Count - 1])
            {
                int lastStart = columns[columns.Count - 1];
                lengths[lengths.Count - 1] = Math.Max(lastStart + lengths[lengths.Count - 1] - 1, end) - lastStart + 1;
                continue;
            }
            rows.Add(run.Row);
            columns.Add(start);
            lengths.Add(end - start + 1);
        }
        return new Region(width, height, rows.ToArray(), columns.ToArray(), lengths.ToArray());
    }
}
