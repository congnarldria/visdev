namespace Vizion;

/// <summary>
/// 區域的圓度、矩形度和方向。圓度是 4π 乘面積除以周長平方。
/// 矩形度是面積除以輪廓最小外接矩形的面積，實心方塊可以大於 1。
/// 方向用像素的二次矩，和 orientation_region 相同。橫條是 0，直條約 1.5708。
/// </summary>
public readonly record struct RegionShape(double Circularity, double Rectangularity, double Orientation, double Ra, double Rb);

public static partial class Operators
{
    public static RegionShape ShapeOf(Region region)
    {
        if (region == null)
            throw new ArgumentNullException(nameof(region));
        int area = region.Area;
        if (area <= 0)
            return new RegionShape(0, 0, 0, 0, 0);
        Moments(region, out double cx, out double cy, out double mu20, out double mu02, out double mu11, out double mass);
        double spread = Math.Sqrt((mu20 - mu02) * (mu20 - mu02) + 4 * mu11 * mu11);
        double major = Math.Max(0, (mu20 + mu02 + spread) / mass);
        double minor = Math.Max(0, (mu20 + mu02 - spread) / mass);
        double phi = mu20 + mu02 < 1e-6 ? 0 : 0.5 * Math.Atan2(2 * mu11, mu20 - mu02);
        var contour = OuterContour(region);
        double length = ContourLength(contour);
        double circularity = length <= 0 ? 0 : Math.Min(1, 4 * Math.PI * area / (length * length));
        double rectArea = MinAreaRect(contour);
        double rectangularity = rectArea > 0 ? area / rectArea : 0;
        return new RegionShape(circularity, rectangularity, phi, Math.Sqrt(2 * major), Math.Sqrt(2 * minor));
    }

    /// <summary>外框。最後一點回到起點。列、行和輪廓相同。</summary>
    public static EdgePoint[] OuterPoints(Region region)
    {
        if (region == null)
            throw new ArgumentNullException(nameof(region));
        List<(int X, int Y)> points = OuterContour(region);
        var outline = new EdgePoint[points.Count];
        for (int i = 0; i < points.Count; i++)
            outline[i] = new EdgePoint(points[i].Y, points[i].X);
        return outline;
    }

    public static double BoundaryLength(Region region)
    {
        if (region == null)
            throw new ArgumentNullException(nameof(region));
        return ContourLength(OuterContour(region));
    }

    static void Moments(Region region, out double cx, out double cy, out double mu20, out double mu02, out double mu11, out double mass)
    {
        double m10 = 0;
        double m01 = 0;
        double count = 0;
        for (int i = 0; i < region.Rows.Length; i++)
        {
            int row = region.Rows[i];
            int column = region.Columns[i];
            int length = region.Lengths[i];
            for (int offset = 0; offset < length; offset++)
            {
                m10 += column + offset;
                m01 += row;
                count++;
            }
        }
        mass = count;
        cx = m10 / count;
        cy = m01 / count;
        mu20 = mu02 = mu11 = 0;
        for (int i = 0; i < region.Rows.Length; i++)
        {
            int row = region.Rows[i];
            int column = region.Columns[i];
            int length = region.Lengths[i];
            double dy = row - cy;
            for (int offset = 0; offset < length; offset++)
            {
                double dx = column + offset - cx;
                mu20 += dx * dx;
                mu02 += dy * dy;
                mu11 += dx * dy;
            }
        }
    }

    static List<(int X, int Y)> OuterContour(Region region)
    {
        int width = region.Width;
        int height = region.Height;
        var foreground = new bool[width * height];
        for (int i = 0; i < region.Rows.Length; i++)
        {
            int runRow = region.Rows[i];
            int runColumn = region.Columns[i];
            int runLength = region.Lengths[i];
            if (runRow < 0 || runRow >= height)
                continue;
            for (int offset = 0; offset < runLength; offset++)
            {
                int runCol = runColumn + offset;
                if (runCol >= 0 && runCol < width)
                    foreground[runRow * width + runCol] = true;
            }
        }
        int start = -1;
        for (int index = 0; index < foreground.Length; index++)
        {
            if (foreground[index])
            {
                start = index;
                break;
            }
        }
        var points = new List<(int X, int Y)>();
        if (start < 0)
            return points;
        int row0 = start / width;
        int col0 = start - row0 * width;
        int[] dRow = { -1, -1, 0, 1, 1, 1, 0, -1 };
        int[] dCol = { 0, 1, 1, 1, 0, -1, -1, -1 };
        int row = row0;
        int col = col0;
        int cameFrom = 6;
        points.Add((col, row));
        for (int guard = 0; guard < width * height * 8; guard++)
        {
            bool found = false;
            for (int turn = 0; turn < 8; turn++)
            {
                int direction = (cameFrom + 1 + turn) % 8;
                int nextRow = row + dRow[direction];
                int nextCol = col + dCol[direction];
                if (nextRow < 0 || nextCol < 0 || nextRow >= height || nextCol >= width)
                    continue;
                if (!foreground[nextRow * width + nextCol])
                    continue;
                row = nextRow;
                col = nextCol;
                cameFrom = (direction + 4) % 8;
                points.Add((col, row));
                found = true;
                break;
            }
            if (!found || (row == row0 && col == col0))
                break;
        }
        return points;
    }

    static double ContourLength(List<(int X, int Y)> points)
    {
        if (points.Count < 2)
            return 0;
        double sum = 0;
        for (int i = 1; i < points.Count; i++)
        {
            double dx = points[i].X - points[i - 1].X;
            double dy = points[i].Y - points[i - 1].Y;
            sum += Math.Sqrt(dx * dx + dy * dy);
        }
        return sum;
    }

    static double MinAreaRect(List<(int X, int Y)> points)
    {
        var unique = new List<(double X, double Y)>();
        var seen = new HashSet<(int X, int Y)>();
        foreach (var point in points)
        {
            if (seen.Add(point))
                unique.Add((point.X, point.Y));
        }
        unique.Sort((left, right) =>
        {
            int byX = left.X.CompareTo(right.X);
            return byX != 0 ? byX : left.Y.CompareTo(right.Y);
        });
        if (unique.Count < 3)
            return 0;
        double Cross((double X, double Y) origin, (double X, double Y) a, (double X, double Y) b) =>
            (a.X - origin.X) * (b.Y - origin.Y) - (a.Y - origin.Y) * (b.X - origin.X);
        var lower = new List<(double X, double Y)>();
        foreach (var point in unique)
        {
            while (lower.Count >= 2 && Cross(lower[lower.Count - 2], lower[lower.Count - 1], point) <= 0)
                lower.RemoveAt(lower.Count - 1);
            lower.Add(point);
        }
        var upper = new List<(double X, double Y)>();
        for (int i = unique.Count - 1; i >= 0; i--)
        {
            var point = unique[i];
            while (upper.Count >= 2 && Cross(upper[upper.Count - 2], upper[upper.Count - 1], point) <= 0)
                upper.RemoveAt(upper.Count - 1);
            upper.Add(point);
        }
        if (lower.Count > 0)
            lower.RemoveAt(lower.Count - 1);
        if (upper.Count > 0)
            upper.RemoveAt(upper.Count - 1);
        lower.AddRange(upper);
        int count = lower.Count;
        if (count < 3)
            return 0;
        double best = double.PositiveInfinity;
        for (int i = 0; i < count; i++)
        {
            var a = lower[i];
            var b = lower[(i + 1) % count];
            double dx = b.X - a.X;
            double dy = b.Y - a.Y;
            double edge = Math.Sqrt(dx * dx + dy * dy);
            if (edge < 1e-12)
                continue;
            double ux = dx / edge;
            double uy = dy / edge;
            double vx = -uy;
            double vy = ux;
            double minU = 0, maxU = 0, minV = 0, maxV = 0;
            for (int j = 0; j < count; j++)
            {
                double du = (lower[j].X - a.X) * ux + (lower[j].Y - a.Y) * uy;
                double dv = (lower[j].X - a.X) * vx + (lower[j].Y - a.Y) * vy;
                if (du < minU) minU = du;
                if (du > maxU) maxU = du;
                if (dv < minV) minV = dv;
                if (dv > maxV) maxV = dv;
            }
            double area = (maxU - minU) * (maxV - minV);
            if (area < best)
                best = area;
        }
        return double.IsPositiveInfinity(best) ? 0 : best;
    }
}
