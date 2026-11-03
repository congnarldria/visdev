namespace Vizion;

public static partial class Operators
{
    /// <summary>兩條邊緣直線的垂直距離。用第一條無限長直線，量到第二條的中點。</summary>
    public static double LineDistance(LineSegment first, LineSegment second)
    {
        double dx = first.ColEnd - first.ColBegin;
        double dy = first.RowEnd - first.RowBegin;
        double length = Math.Sqrt(dx * dx + dy * dy);
        if (length < 1e-9)
            throw new ArgumentException("第一條直線太短，量不出距離");
        double midColumn = (second.ColBegin + second.ColEnd) * 0.5;
        double midRow = (second.RowBegin + second.RowEnd) * 0.5;
        double cross = dx * (midRow - first.RowBegin) - dy * (midColumn - first.ColBegin);
        return Math.Abs(cross) / length;
    }
}
