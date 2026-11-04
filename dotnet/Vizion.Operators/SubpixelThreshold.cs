namespace Vizion;

/// <summary>次像素二值化的結果。物件是八連通，輪廓沿著灰階等於門檻的位置。</summary>
public sealed class SubpixFigure
{
    public SubpixFigure(Region[] objects, EdgePoint[][] contours, int[] groups)
    {
        if (objects == null || contours == null || groups == null)
            throw new ArgumentNullException(nameof(objects));
        if (contours.Length != groups.Length)
            throw new ArgumentException("輪廓和分組要一樣長");
        Objects = objects;
        Contours = contours;
        Groups = groups;
    }

    public Region[] Objects { get; }
    public EdgePoint[][] Contours { get; }
    public int[] Groups { get; }

    public static Overlay[] Overlays(SubpixFigure figure, int stroke)
    {
        if (figure == null)
            throw new ArgumentNullException(nameof(figure));
        int width = stroke < 1 ? 1 : stroke > 10 ? 10 : stroke;
        var overlays = new List<Overlay>(figure.Objects.Length + figure.Contours.Length);
        bool multi = figure.Objects.Length > 1;
        for (int i = 0; i < figure.Objects.Length; i++)
        {
            uint color = Operators.Colors[(multi ? i : 0) % Operators.Colors.Length];
            overlays.Add(figure.Objects[i].ToOverlay((color & 0x00FFFFFFu) | (110u << 24)));
        }
        for (int i = 0; i < figure.Contours.Length; i++)
        {
            EdgePoint[] chain = figure.Contours[i];
            if (chain.Length < 2)
                continue;
            uint color = Operators.Colors[figure.Groups[i] % Operators.Colors.Length];
            var rows = new double[chain.Length];
            var columns = new double[chain.Length];
            for (int p = 0; p < chain.Length; p++)
            {
                rows[p] = chain[p].Row;
                columns[p] = chain[p].Column;
            }
            overlays.Add(Overlay.Polyline(rows, columns, color).WithStroke(width));
        }
        return overlays.ToArray();
    }
}

public static partial class Operators
{
    /// <summary>沿著灰階等於 Threshold 的位置取出次像素輪廓，亮處留在輪廓內。結果是 XLD。會自動用八連通拆開，每個物件一種顏色，點一個只框那一個，和 connection 距離 1 一樣。門檻若是數字，可以在統計圖拖曳，也可以用次像素視窗調，那個視窗不會擋住主視窗。</summary>
    /// <remarks>次像素二值化</remarks>
    /// <example>
    /// <code>Vizion.threshold_sub_pix (Image, Border, 128)</code>
    /// </example>
    /// <param name="image">灰階或彩色影像。</param>
    /// <param name="level">灰階交界，0 到 255。</param>
    /// <returns>次像素輪廓。亮處留在輪廓內。</returns>
    public static SubpixFigure ThresholdSubPix(ImageBuffer image, double level)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (level < 0 || level > 255)
            throw new ArgumentOutOfRangeException(nameof(level), "Threshold 要在 0 到 255");
        int minGray = (int)Math.Ceiling(level);
        if (minGray > 255)
            return new SubpixFigure(Array.Empty<Region>(), Array.Empty<EdgePoint[]>(), Array.Empty<int>());
        Region mask = Threshold(image, minGray, 255);
        Region[] parts = Connection(mask);
        if (parts.Length == 0)
            return new SubpixFigure(Array.Empty<Region>(), Array.Empty<EdgePoint[]>(), Array.Empty<int>());

        int width = image.Width;
        int height = image.Height;
        byte[] gray = image.GrayBytes();
        var labels = new int[width * height];
        for (int part = 0; part < parts.Length; part++)
            PaintLabel(labels, width, height, parts[part], part + 1);

        var buckets = new List<List<(int X, int Y)>>[parts.Length];
        for (int i = 0; i < buckets.Length; i++)
            buckets[i] = new List<List<(int X, int Y)>>();
        var traced = new bool[width * height];
        for (int y = 0; y < height; y++)
        {
            int row = y * width;
            for (int x = 0; x < width; x++)
            {
                int index = row + x;
                int id = labels[index];
                if (id == 0 || traced[index])
                    continue;
                bool leftBackground = x == 0 || labels[index - 1] != id;
                bool rightBackground = x + 1 >= width || labels[index + 1] != id;
                if (!leftBackground && !rightBackground)
                    continue;
                List<(int X, int Y)> chain = Trace(labels, width, height, x, y, id, leftBackground ? 4 : 0);
                buckets[id - 1].Add(chain);
                for (int p = 0; p < chain.Count; p++)
                    traced[chain[p].Y * width + chain[p].X] = true;
            }
        }

        var objects = new List<Region>();
        var contours = new List<EdgePoint[]>();
        var groups = new List<int>();
        for (int part = 0; part < parts.Length && objects.Count < 400; part++)
        {
            int before = contours.Count;
            foreach (List<(int X, int Y)> chain in buckets[part])
            {
                EdgePoint[] refined = Refine(chain, gray, width, height, level);
                if (refined.Length >= 3)
                    contours.Add(refined);
            }
            if (contours.Count == before)
                continue;
            int group = objects.Count;
            objects.Add(parts[part]);
            for (int i = before; i < contours.Count; i++)
                groups.Add(group);
        }
        return new SubpixFigure(objects.ToArray(), contours.ToArray(), groups.ToArray());
    }

    static void PaintLabel(int[] labels, int width, int height, Region region, int id)
    {
        for (int i = 0; i < region.Rows.Length; i++)
        {
            int row = region.Rows[i];
            int column = region.Columns[i];
            int length = region.Lengths[i];
            if ((uint)row >= (uint)height || length <= 0)
                continue;
            if (column < 0)
            {
                length += column;
                column = 0;
            }
            if (column >= width)
                continue;
            if (column + length > width)
                length = width - column;
            int offset = row * width + column;
            for (int k = 0; k < length; k++)
                labels[offset + k] = id;
        }
    }

    static readonly int[] StepX = { 1, 1, 0, -1, -1, -1, 0, 1 };
    static readonly int[] StepY = { 0, -1, -1, -1, 0, 1, 1, 1 };

    static List<(int X, int Y)> Trace(int[] labels, int width, int height, int x, int y, int id, int previous)
    {
        var chain = new List<(int X, int Y)>();
        int cx = x;
        int cy = y;
        int guard = width * height * 8 + 4;
        for (int n = 0; n < guard; n++)
        {
            chain.Add((cx, cy));
            int found = -1;
            int nextX = cx;
            int nextY = cy;
            for (int k = 0; k < 8; k++)
            {
                int direction = (previous + 1 + k) % 8;
                int tx = cx + StepX[direction];
                int ty = cy + StepY[direction];
                if ((uint)tx < (uint)width && (uint)ty < (uint)height && labels[ty * width + tx] == id)
                {
                    found = direction;
                    nextX = tx;
                    nextY = ty;
                    break;
                }
            }
            if (found < 0)
                break;
            previous = (found + 4) % 8;
            cx = nextX;
            cy = nextY;
            if (cx == x && cy == y)
                break;
        }
        return chain;
    }

    static EdgePoint[] Refine(List<(int X, int Y)> raw, byte[] gray, int width, int height, double level)
    {
        int countRaw = raw.Count;
        if (countRaw < 3)
            return ToPoints(raw, countRaw);
        double dx = raw[0].X - raw[countRaw - 1].X;
        double dy = raw[0].Y - raw[countRaw - 1].Y;
        bool closed = dx * dx + dy * dy < 2.25;
        int count = closed ? countRaw - 1 : countRaw;
        var rows = new double[count + (closed ? 1 : 0)];
        var columns = new double[rows.Length];
        for (int index = 0; index < count; index++)
        {
            (int X, int Y) point = raw[index];
            (int X, int Y) prev = raw[Mod(index - 1, count)];
            (int X, int Y) next = raw[Mod(index + 1, count)];
            double tangentX = next.X - prev.X;
            double tangentY = next.Y - prev.Y;
            double norm = Math.Sqrt(tangentX * tangentX + tangentY * tangentY);
            double offset = 0;
            double normalX = 0;
            double normalY = 0;
            if (norm >= 1e-6)
            {
                normalX = -tangentY / norm;
                normalY = tangentX / norm;
                double left = Sample(gray, width, height, point.X - normalX, point.Y - normalY);
                double mid = Sample(gray, width, height, point.X, point.Y);
                double right = Sample(gray, width, height, point.X + normalX, point.Y + normalY);
                if ((left - level) * (mid - level) <= 0 && left != mid)
                    offset = -1 + (level - left) / (mid - left);
                else if ((mid - level) * (right - level) <= 0 && mid != right)
                    offset = (level - mid) / (right - mid);
            }
            columns[index] = point.X + normalX * offset;
            rows[index] = point.Y + normalY * offset;
        }
        if (closed)
        {
            columns[count] = columns[0];
            rows[count] = rows[0];
        }
        var points = new EdgePoint[rows.Length];
        for (int i = 0; i < points.Length; i++)
            points[i] = new EdgePoint(rows[i], columns[i]);
        return points;
    }

    static EdgePoint[] ToPoints(List<(int X, int Y)> raw, int count)
    {
        var points = new EdgePoint[count];
        for (int i = 0; i < count; i++)
            points[i] = new EdgePoint(raw[i].Y, raw[i].X);
        return points;
    }

    static int Mod(int value, int count)
    {
        int remainder = value % count;
        return remainder < 0 ? remainder + count : remainder;
    }

    static double Sample(byte[] gray, int width, int height, double x, double y)
    {
        if (x < 0 || y < 0 || x > width - 1 || y > height - 1)
        {
            if (x < 0)
                x = 0;
            else if (x > width - 1)
                x = width - 1;
            if (y < 0)
                y = 0;
            else if (y > height - 1)
                y = height - 1;
        }
        int x0 = (int)Math.Floor(x);
        int y0 = (int)Math.Floor(y);
        int x1 = x0 + 1 < width ? x0 + 1 : width - 1;
        int y1 = y0 + 1 < height ? y0 + 1 : height - 1;
        double dx = x - x0;
        double dy = y - y0;
        double top = gray[y0 * width + x0] * (1 - dx) + gray[y0 * width + x1] * dx;
        double bottom = gray[y1 * width + x0] * (1 - dx) + gray[y1 * width + x1] * dx;
        return top * (1 - dy) + bottom * dy;
    }
}
