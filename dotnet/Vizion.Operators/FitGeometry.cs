namespace Vizion;

/// <summary>
/// 找線、找圓。點是 (行, 列)，和 Vizion.fit_line_contour_xld、fit_circle_contour_xld 相同。
/// 離模型太遠的飛點會丟掉再擬一次。點都在模型上時，結果和一次擬合一樣。
/// </summary>
public readonly record struct LineSegment(double RowBegin, double ColBegin, double RowEnd, double ColEnd);

public readonly record struct CircleFit(double Row, double Column, double Radius);

public static partial class Operators
{
    public static LineSegment FitLine(IReadOnlyList<(double Row, double Col)> points)
    {
        if (points == null || points.Count < 2)
            throw new ArgumentException("直線擬合至少要 2 個點");
        var cols = new double[points.Count];
        var rows = new double[points.Count];
        for (int i = 0; i < points.Count; i++)
        {
            cols[i] = points[i].Col;
            rows[i] = points[i].Row;
        }
        return FitLineRaw(cols, rows);
    }

    public static CircleFit FitCircle(IReadOnlyList<(double Row, double Col)> points)
    {
        if (points == null || points.Count < 3)
            throw new ArgumentException("圓擬合至少要 3 個點");
        var cols = new double[points.Count];
        var rows = new double[points.Count];
        for (int i = 0; i < points.Count; i++)
        {
            cols[i] = points[i].Col;
            rows[i] = points[i].Row;
        }
        var (row, col, radius) = Pratt(cols, rows, points.Count);
        var keep = new bool[points.Count];
        for (int step = 0; step < 4; step++)
        {
            var distance = new double[points.Count];
            for (int i = 0; i < points.Count; i++)
                distance[i] = Math.Abs(Hypot(cols[i] - col, rows[i] - row) - radius);
            double typical = Median(distance);
            double far = Max(distance);
            if (typical < 1e-8 && far < 1e-6)
                break;
            double limit = typical > 0.4 ? 2.5 * typical : 1.0;
            int alive = 0;
            for (int i = 0; i < points.Count; i++)
            {
                keep[i] = distance[i] <= limit;
                if (keep[i]) alive++;
            }
            if (alive < 3)
                break;
            (row, col, radius) = Pratt(cols, rows, keep);
        }
        return new CircleFit(row, col, radius);
    }

    static LineSegment FitLineRaw(double[] cols, double[] rows)
    {
        int count = cols.Length;
        var weight = new double[count];
        for (int i = 0; i < count; i++) weight[i] = 1;
        double centerCol = Mean(cols, weight);
        double centerRow = Mean(rows, weight);
        var (dirCol, dirRow) = Principal(cols, rows, centerCol, centerRow, weight);
        for (int step = 0; step < 4; step++)
        {
            double normalCol = -dirRow;
            double normalRow = dirCol;
            var distance = new double[count];
            for (int i = 0; i < count; i++)
            {
                double dc = cols[i] - centerCol;
                double dr = rows[i] - centerRow;
                distance[i] = Math.Abs(dc * normalCol + dr * normalRow);
            }
            double typical = Median(distance);
            double far = Max(distance);
            if (typical < 1e-8 && far < 1e-6)
                break;
            double limit = typical > 0.4 ? 2.5 * typical : 1.0;
            double sum = 0;
            for (int i = 0; i < count; i++)
            {
                weight[i] = distance[i] <= limit ? 1 : 0;
                sum += weight[i];
            }
            if (sum < 2)
                break;
            centerCol = Mean(cols, weight);
            centerRow = Mean(rows, weight);
            (dirCol, dirRow) = Principal(cols, rows, centerCol, centerRow, weight);
        }
        double min = double.PositiveInfinity;
        double max = double.NegativeInfinity;
        int kept = 0;
        for (int i = 0; i < count; i++)
        {
            if (weight[i] <= 0.05) continue;
            kept++;
            double dc = cols[i] - centerCol;
            double dr = rows[i] - centerRow;
            double projection = dc * dirCol + dr * dirRow;
            if (projection < min) min = projection;
            if (projection > max) max = projection;
        }
        if (kept < 2)
        {
            min = double.PositiveInfinity;
            max = double.NegativeInfinity;
            for (int i = 0; i < count; i++)
            {
                double dc = cols[i] - centerCol;
                double dr = rows[i] - centerRow;
                double projection = dc * dirCol + dr * dirRow;
                if (projection < min) min = projection;
                if (projection > max) max = projection;
            }
        }
        return new LineSegment(
            centerRow + dirRow * min,
            centerCol + dirCol * min,
            centerRow + dirRow * max,
            centerCol + dirCol * max);
    }

    static (double Col, double Row) Principal(double[] cols, double[] rows, double centerCol, double centerRow, double[] weight)
    {
        double xx = 0, yy = 0, xy = 0;
        for (int i = 0; i < cols.Length; i++)
        {
            double scale = Math.Sqrt(weight[i]);
            double dc = (cols[i] - centerCol) * scale;
            double dr = (rows[i] - centerRow) * scale;
            xx += dc * dc;
            yy += dr * dr;
            xy += dc * dr;
        }
        double gap = xx - yy;
        double larger = 0.5 * (xx + yy + Math.Sqrt(gap * gap + 4 * xy * xy));
        double dirCol;
        double dirRow;
        if (Math.Abs(xy) > 1e-15)
        {
            dirCol = larger - yy;
            dirRow = xy;
        }
        else if (xx >= yy)
        {
            dirCol = 1;
            dirRow = 0;
        }
        else
        {
            dirCol = 0;
            dirRow = 1;
        }
        double length = Hypot(dirCol, dirRow);
        if (length < 1e-15)
            return (1, 0);
        return (dirCol / length, dirRow / length);
    }

    static (double Row, double Col, double Radius) Pratt(double[] cols, double[] rows, int count)
    {
        var keep = new bool[count];
        for (int i = 0; i < count; i++) keep[i] = true;
        return Pratt(cols, rows, keep);
    }

    static (double Row, double Col, double Radius) Pratt(double[] cols, double[] rows, bool[] keep)
    {
        int count = 0;
        double sumCol = 0, sumRow = 0;
        for (int i = 0; i < keep.Length; i++)
        {
            if (!keep[i]) continue;
            count++;
            sumCol += cols[i];
            sumRow += rows[i];
        }
        if (count < 3)
            throw new InvalidOperationException("這些點擬合不出圓");
        double originCol = sumCol / count;
        double originRow = sumRow / count;
        double momentCol = 0, momentRow = 0, momentCross = 0;
        double momentColSq = 0, momentRowSq = 0, momentSq = 0;
        for (int i = 0; i < keep.Length; i++)
        {
            if (!keep[i]) continue;
            double shiftedCol = cols[i] - originCol;
            double shiftedRow = rows[i] - originRow;
            double square = shiftedCol * shiftedCol + shiftedRow * shiftedRow;
            momentCol += shiftedCol * shiftedCol;
            momentRow += shiftedRow * shiftedRow;
            momentCross += shiftedCol * shiftedRow;
            momentColSq += shiftedCol * square;
            momentRowSq += shiftedRow * square;
            momentSq += square * square;
        }
        momentCol /= count;
        momentRow /= count;
        momentCross /= count;
        momentColSq /= count;
        momentRowSq /= count;
        momentSq /= count;
        double momentSum = momentCol + momentRow;
        double covariance = momentCol * momentRow - momentCross * momentCross;
        double cubic = 4 * momentSum;
        double quad = -3 * momentSum * momentSum - momentSq;
        double linear = momentSq * momentSum + 4 * covariance * momentSum
            - momentColSq * momentColSq - momentRowSq * momentRowSq - momentSum * momentSum * momentSum;
        double constant = momentColSq * momentColSq * momentRow + momentRowSq * momentRowSq * momentCol
            - momentSq * covariance - 2 * momentColSq * momentRowSq * momentCross
            + momentSum * momentSum * covariance;
        double root = 0;
        double value = 1e20;
        for (int step = 0; step < 20; step++)
        {
            double previous = value;
            value = constant + root * (linear + root * (quad + root * cubic));
            if (Math.Abs(value) > Math.Abs(previous))
            {
                root = 0;
                break;
            }
            double slope = linear + root * (2 * quad + 3 * root * cubic);
            if (Math.Abs(slope) < 1e-18)
                break;
            double updated = root - value / slope;
            if (updated < 0)
            {
                root = 0;
                break;
            }
            if (Math.Abs(updated) > 1e-12 && Math.Abs((updated - root) / updated) < 1e-12)
            {
                root = updated;
                break;
            }
            root = updated;
        }
        double determinant = root * root - root * momentSum + covariance;
        if (Math.Abs(determinant) < 1e-15)
            throw new InvalidOperationException("這些點擬合不出圓");
        double centerCol = (momentColSq * (momentRow - root) - momentRowSq * momentCross) / determinant / 2;
        double centerRow = (momentRowSq * (momentCol - root) - momentColSq * momentCross) / determinant / 2;
        double radius2 = centerCol * centerCol + centerRow * centerRow + momentSum + 2 * root;
        if (radius2 <= 1e-8)
            throw new InvalidOperationException("這些點擬合不出圓");
        return (centerRow + originRow, centerCol + originCol, Math.Sqrt(radius2));
    }

    static double Hypot(double x, double y) => Math.Sqrt(x * x + y * y);

    static double Mean(double[] values, double[] weight)
    {
        double sum = 0, mass = 0;
        for (int i = 0; i < values.Length; i++)
        {
            sum += weight[i] * values[i];
            mass += weight[i];
        }
        return mass == 0 ? 0 : sum / mass;
    }

    static double Median(double[] values)
    {
        var copy = (double[])values.Clone();
        Array.Sort(copy);
        int mid = copy.Length / 2;
        if (copy.Length % 2 == 1)
            return copy[mid];
        return 0.5 * (copy[mid - 1] + copy[mid]);
    }

    static double Max(double[] values)
    {
        double far = values[0];
        for (int i = 1; i < values.Length; i++)
            if (values[i] > far) far = values[i];
        return far;
    }
}
