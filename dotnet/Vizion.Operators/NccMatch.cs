namespace Vizion;

/// <summary>
/// 簡化的正規相關。教導框中心是原點。
/// 先在高斯金字塔的粗圖上搜尋角度和位置，再回到原圖用分數曲面的拋物線修列和行。
/// 這不是 find_ncc_model，也不使用 visdev_match.pyd。分數是灰階的相關係數。
/// ignoreGlobalPolarity 為真時取絕對值，黑白對調仍算對上。
/// </summary>
public readonly record struct NccHit(double Row, double Column, double Angle, double Score);

public static partial class Operators
{
    public static NccHit[] FindNcc(
        ImageBuffer scene,
        ImageBuffer template,
        int row1,
        int column1,
        int row2,
        int column2,
        double minScore = 0.5,
        bool ignoreGlobalPolarity = false)
    {
        if (scene == null || template == null)
            throw new ArgumentNullException(nameof(scene));
        if (row2 < row1 || column2 < column1)
            throw new ArgumentOutOfRangeException(nameof(row2));
        var sceneLevels = BytePyramid(scene.GrayBytes(), scene.Width, scene.Height, 3, stopWhenSmall: true);
        var templateLevels = BytePyramid(template.GrayBytes(), template.Width, template.Height, sceneLevels.Count, stopWhenSmall: false);
        if (templateLevels.Count != sceneLevels.Count)
            throw new InvalidOperationException("樣板和場景的金字塔層數不一致。");
        var levels = new List<NccLevel>();
        for (int i = 0; i < sceneLevels.Count; i++)
        {
            var image = sceneLevels[i];
            var patch = templateLevels[i];
            var points = BoxPoints(patch.Gray, patch.Width, patch.Height, row1, column1, row2, column2, image.Scale, i == 0 ? 700 : 360);
            levels.Add(new NccLevel(image.Width, image.Height, image.Scale, ToFloat(image.Gray), points));
        }
        if (levels[0].Points.Length < 16)
            throw new InvalidOperationException("教導框太小。請把框拉大。");

        int coarseIndex = levels.Count - 1;
        var coarse = levels[coarseIndex];
        var hits = new List<NccHit>();
        const double angleStep = 4.0 * Math.PI / 180.0;
        int margin = 6;
        double coarseMin = Math.Min(minScore, minScore * 0.72 + 0.05);
        for (double angle = -Math.PI; angle < Math.PI - 1e-6; angle += angleStep)
        {
            RotateNcc(coarse.Points, angle, out float[] rx, out float[] ry, out float[] val);
            for (int row = margin; row < coarse.Height - margin; row += 2)
            {
                for (int col = margin; col < coarse.Width - margin; col += 2)
                {
                    double score = Correlate(coarse.Gray, coarse.Width, coarse.Height, rx, ry, val, col, row, ignoreGlobalPolarity);
                    if (score >= coarseMin)
                        hits.Add(new NccHit(row * coarse.Scale, col * coarse.Scale, angle, score));
                }
            }
        }

        var kept = SuppressNcc(hits, 70, 16);
        for (int index = coarseIndex - 1; index >= 0; index--)
        {
            var field = levels[index];
            double angleWindow = index == 0 ? 2.0 * Math.PI / 180.0 : angleStep;
            var promoted = new List<NccHit>();
            foreach (var hit in kept)
            {
                var best = PromoteNcc(hit, field, angleWindow, ignoreGlobalPolarity);
                if (best.Score >= minScore * 0.75)
                    promoted.Add(best);
            }
            kept = SuppressNcc(promoted, 70, 8);
        }

        var refined = new List<NccHit>();
        var full = levels[0];
        foreach (var hit in kept)
        {
            if (hit.Score < minScore * 0.75)
                continue;
            var posed = SnapNcc(hit, full, ignoreGlobalPolarity);
            posed = PolishNccAngle(posed, full, ignoreGlobalPolarity);
            posed = SnapNcc(posed, full, ignoreGlobalPolarity);
            posed = ParabolicNcc(posed, full, ignoreGlobalPolarity);
            posed = posed with { Angle = WrapAngle(posed.Angle) };
            if (posed.Score >= minScore)
                refined.Add(posed);
        }
        return SuppressNcc(refined, 70, 8).ToArray();
    }

    readonly record struct NccPoint(float X, float Y, float Value);

    readonly record struct GrayLevel(int Width, int Height, int Scale, byte[] Gray);

    readonly record struct NccLevel(int Width, int Height, int Scale, float[] Gray, NccPoint[] Points);

    static List<GrayLevel> BytePyramid(byte[] gray, int width, int height, int maxLevels, bool stopWhenSmall)
    {
        var levels = new List<GrayLevel>();
        int scale = 1;
        for (int level = 0; level < maxLevels; level++)
        {
            levels.Add(new GrayLevel(width, height, scale, gray));
            if (level + 1 == maxLevels || width < 2 || height < 2)
                break;
            if (stopWhenSmall && (width < 48 || height < 48))
                break;
            PyrDown(gray, width, height, out gray, out width, out height);
            scale *= 2;
        }
        return levels;
    }

    static float[] ToFloat(byte[] gray)
    {
        var values = new float[gray.Length];
        for (int i = 0; i < gray.Length; i++)
            values[i] = gray[i];
        return values;
    }

    static NccPoint[] BoxPoints(byte[] gray, int width, int height, int row1, int column1, int row2, int column2, int scale, int limit)
    {
        double originRow = (row1 + row2) / 2.0 / scale;
        double originCol = (column1 + column2) / 2.0 / scale;
        int top = Math.Max(1, (int)Math.Floor(row1 / (double)scale));
        int left = Math.Max(1, (int)Math.Floor(column1 / (double)scale));
        int bottom = Math.Min(height - 2, (int)Math.Ceiling(row2 / (double)scale));
        int right = Math.Min(width - 2, (int)Math.Ceiling(column2 / (double)scale));
        var points = new List<NccPoint>();
        if (bottom < top || right < left)
            return Array.Empty<NccPoint>();
        int rows = bottom - top + 1;
        int cols = right - left + 1;
        int stride = Math.Max(1, (int)Math.Ceiling(Math.Sqrt(rows * cols / (double)Math.Max(limit, 1))));
        for (int row = top; row <= bottom; row += stride)
        {
            for (int col = left; col <= right; col += stride)
            {
                points.Add(new NccPoint((float)(col - originCol), (float)(row - originRow), gray[row * width + col]));
            }
        }
        return points.ToArray();
    }

    static void RotateNcc(NccPoint[] points, double angle, out float[] x, out float[] y, out float[] value)
    {
        float cosine = (float)Math.Cos(angle);
        float sine = (float)Math.Sin(angle);
        x = new float[points.Length];
        y = new float[points.Length];
        value = new float[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            x[i] = points[i].X * cosine + points[i].Y * sine;
            y[i] = -points[i].X * sine + points[i].Y * cosine;
            value[i] = points[i].Value;
        }
    }

    static NccHit PromoteNcc(NccHit hit, NccLevel field, double angleWindow, bool absolute)
    {
        var best = new NccHit(hit.Row, hit.Column, hit.Angle, -1);
        double baseRow = hit.Row / field.Scale;
        double baseCol = hit.Column / field.Scale;
        for (int angleStep = -2; angleStep <= 2; angleStep++)
        {
            double angle = hit.Angle + angleStep * (angleWindow / 2.0);
            RotateNcc(field.Points, angle, out float[] rx, out float[] ry, out float[] val);
            for (int dRow = -2; dRow <= 2; dRow++)
            {
                for (int dCol = -2; dCol <= 2; dCol++)
                {
                    double row = baseRow + dRow;
                    double col = baseCol + dCol;
                    if (row < 2 || col < 2 || row >= field.Height - 2 || col >= field.Width - 2)
                        continue;
                    double score = Correlate(field.Gray, field.Width, field.Height, rx, ry, val, col, row, absolute);
                    if (score > best.Score)
                        best = new NccHit(row * field.Scale, col * field.Scale, angle, score);
                }
            }
        }
        return best.Score < 0 ? hit : best;
    }

    static NccHit SnapNcc(NccHit hit, NccLevel field, bool absolute)
    {
        int row0 = (int)Math.Round(hit.Row);
        int col0 = (int)Math.Round(hit.Column);
        RotateNcc(field.Points, hit.Angle, out float[] rx, out float[] ry, out float[] val);
        var scored = new List<(double Row, double Col, double Score)>();
        double best = hit.Score;
        for (int dRow = -2; dRow <= 2; dRow++)
        {
            for (int dCol = -2; dCol <= 2; dCol++)
            {
                double row = row0 + dRow * 0.5;
                double col = col0 + dCol * 0.5;
                if (row < 2 || col < 2 || row >= field.Height - 2 || col >= field.Width - 2)
                    continue;
                double score = Correlate(field.Gray, field.Width, field.Height, rx, ry, val, col, row, absolute);
                if (score > best)
                    best = score;
                scored.Add((row, col, score));
            }
        }
        double rowSum = 0;
        double colSum = 0;
        int tied = 0;
        foreach (var item in scored)
        {
            if (best - item.Score > 0.002)
                continue;
            rowSum += item.Row;
            colSum += item.Col;
            tied++;
        }
        if (tied == 0)
            return hit;
        double rowMean = rowSum / tied;
        double colMean = colSum / tied;
        double scoreMean = Correlate(field.Gray, field.Width, field.Height, rx, ry, val, colMean, rowMean, absolute);
        if (tied == 1 || scoreMean + 0.002 < best)
        {
            foreach (var item in scored)
            {
                if (item.Score == best)
                    return new NccHit(item.Row, item.Col, hit.Angle, item.Score);
            }
        }
        return new NccHit(rowMean, colMean, hit.Angle, scoreMean);
    }

    static NccHit PolishNccAngle(NccHit hit, NccLevel field, bool absolute)
    {
        var best = hit;
        double bestSupport = -1;
        double step = 0.5 * Math.PI / 180.0;
        for (int index = -6; index <= 6; index++)
        {
            double angle = hit.Angle + index * step;
            RotateNcc(field.Points, angle, out float[] rx, out float[] ry, out float[] val);
            double support = Support(field, rx, ry, val, hit.Column, hit.Row, absolute);
            if (support > bestSupport)
            {
                bestSupport = support;
                double score = Correlate(field.Gray, field.Width, field.Height, rx, ry, val, hit.Column, hit.Row, absolute);
                best = new NccHit(hit.Row, hit.Column, angle, score);
            }
        }
        return best;
    }

    static double Support(NccLevel field, float[] x, float[] y, float[] value, double column, double row, bool absolute)
    {
        double sum =
            Correlate(field.Gray, field.Width, field.Height, x, y, value, column, row, absolute)
            + Correlate(field.Gray, field.Width, field.Height, x, y, value, column + 0.5, row, absolute)
            + Correlate(field.Gray, field.Width, field.Height, x, y, value, column - 0.5, row, absolute)
            + Correlate(field.Gray, field.Width, field.Height, x, y, value, column, row + 0.5, absolute)
            + Correlate(field.Gray, field.Width, field.Height, x, y, value, column, row - 0.5, absolute);
        return sum / 5.0;
    }

    static NccHit ParabolicNcc(NccHit hit, NccLevel field, bool absolute)
    {
        double row = hit.Row;
        double col = hit.Column;
        double angle = hit.Angle;
        double step = 0.8 * Math.PI / 180.0;
        const double spacing = 0.5;
        for (int pass = 0; pass < 2; pass++)
        {
            RotateNcc(field.Points, angle, out float[] rx, out float[] ry, out float[] val);
            double center = Correlate(field.Gray, field.Width, field.Height, rx, ry, val, col, row, absolute);
            double At(double c, double r, float[] px, float[] py, float[] pv) =>
                Correlate(field.Gray, field.Width, field.Height, px, py, pv, c, r, absolute);
            double dRow = Parabola(At(col, row - spacing, rx, ry, val), center, At(col, row + spacing, rx, ry, val));
            double dCol = Parabola(At(col - spacing, row, rx, ry, val), center, At(col + spacing, row, rx, ry, val));
            RotateNcc(field.Points, angle - step, out float[] rx0, out float[] ry0, out float[] val0);
            RotateNcc(field.Points, angle + step, out float[] rx1, out float[] ry1, out float[] val1);
            double dAngle = Parabola(
                Correlate(field.Gray, field.Width, field.Height, rx0, ry0, val0, col, row, absolute),
                center,
                Correlate(field.Gray, field.Width, field.Height, rx1, ry1, val1, col, row, absolute));
            row += dRow * spacing;
            col += dCol * spacing;
            angle += dAngle * step;
            step *= 0.5;
        }
        RotateNcc(field.Points, angle, out float[] fx, out float[] fy, out float[] fv);
        double score = Correlate(field.Gray, field.Width, field.Height, fx, fy, fv, col, row, absolute);
        if (score + 1e-4 < hit.Score)
            return hit;
        return new NccHit(row, col, angle, score);
    }

    static double Correlate(float[] gray, int width, int height, float[] x, float[] y, float[] value, double column, double row, bool absolute = false)
    {
        int n = x.Length;
        int minCount = Math.Max(16, (int)(n * 0.65));
        double sumS = 0, sumT = 0, sumSS = 0, sumTT = 0, sumST = 0;
        int count = 0;
        for (int i = 0; i < n; i++)
        {
            if (!Sample(gray, width, height, column + x[i], row + y[i], out float sample))
                continue;
            float template = value[i];
            sumS += sample;
            sumT += template;
            sumSS += (double)sample * sample;
            sumTT += (double)template * template;
            sumST += (double)sample * template;
            count++;
        }
        if (count < minCount)
            return 0;
        double varS = sumSS - sumS * sumS / count;
        double varT = sumTT - sumT * sumT / count;
        double cov = sumST - sumS * sumT / count;
        if (varS < 1e-6 || varT < 1e-6)
            return 0;
        double score = cov / Math.Sqrt(varS * varT);
        return absolute ? Math.Abs(score) : score;
    }

    static List<NccHit> SuppressNcc(List<NccHit> hits, double minDistance, int maximum)
    {
        var kept = new List<NccHit>();
        double limit = minDistance * minDistance;
        foreach (var hit in hits.OrderByDescending(item => item.Score))
        {
            bool near = false;
            foreach (var other in kept)
            {
                double dRow = other.Row - hit.Row;
                double dCol = other.Column - hit.Column;
                if (dRow * dRow + dCol * dCol < limit)
                {
                    near = true;
                    break;
                }
            }
            if (!near)
                kept.Add(hit);
            if (kept.Count == maximum)
                break;
        }
        return kept;
    }
}
