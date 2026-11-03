namespace Vizion;

/// <summary>
/// 簡化的正規相關。教導框中心是原點。
/// 先在高斯金字塔的粗圖上搜尋角度和位置，再回到原圖。
/// 分數用 Keys 三次內插。平坦的峰留在取樣點上。接著沿灰階梯度做 Lucas-Kanade。
/// 分數沒有變好就不採用。Lucas-Kanade 不改角度。
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
            posed = InterpolateNcc(posed, full, ignoreGlobalPolarity);
            var fitted = LucasKanade(posed, full, ignoreGlobalPolarity);
            if (fitted.Score + 1e-4 >= posed.Score)
                posed = fitted;
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

    static NccHit InterpolateNcc(NccHit hit, NccLevel field, bool absolute)
    {
        double row = hit.Row;
        double col = hit.Column;
        double angle = hit.Angle;
        double step = 0.8 * Math.PI / 180.0;
        row = AxisPeak(field, absolute, angle, col, row, alongRow: true, spacing: 1);
        col = AxisPeak(field, absolute, angle, col, row, alongRow: false, spacing: 1);
        angle = AxisPeak(field, absolute, angle, col, row, alongRow: true, spacing: step, angleAxis: true);
        RotateNcc(field.Points, angle, out float[] fx, out float[] fy, out float[] fv);
        double score = Correlate(field.Gray, field.Width, field.Height, fx, fy, fv, col, row, absolute);
        if (score + 1e-4 < hit.Score)
            return hit;
        return new NccHit(row, col, angle, score);
    }

    static double AxisPeak(NccLevel field, bool absolute, double angle, double column, double row, bool alongRow, double spacing, bool angleAxis = false)
    {
        double[] At(double origin)
        {
            var samples = new double[5];
            for (int index = 0; index < 5; index++)
            {
                double delta = (index - 2) * spacing;
                double sampleRow = row;
                double sampleCol = column;
                double sampleAngle = angle;
                if (angleAxis)
                    sampleAngle = origin + delta;
                else if (alongRow)
                    sampleRow = origin + delta;
                else
                    sampleCol = origin + delta;
                RotateNcc(field.Points, sampleAngle, out float[] rx, out float[] ry, out float[] val);
                samples[index] = Correlate(field.Gray, field.Width, field.Height, rx, ry, val, sampleCol, sampleRow, absolute);
            }
            return samples;
        }
        double center = angleAxis ? angle : alongRow ? row : column;
        double[] window = At(center);
        double offset = KeysPeak(window);
        if (Math.Abs(offset) > 1)
        {
            center += offset < 0 ? -spacing : spacing;
            offset = KeysPeak(At(center));
            if (offset < -1)
                offset = -1;
            if (offset > 1)
                offset = 1;
        }
        return center + offset * spacing;
    }

    static double KeysWeight(double distance)
    {
        double span = Math.Abs(distance);
        if (span < 1)
            return ((1.5 * span - 2.5) * span * span) + 1;
        if (span < 2)
            return ((-0.5 * span + 2.5) * span - 4) * span + 2;
        return 0;
    }

    /// <summary>五個相鄰分數上的 Keys 三次峰。索引 2 是目前的取樣點。平坦就留在原點。</summary>
    static double KeysPeak(double[] samples)
    {
        double peak = samples[0];
        double lowest = samples[0];
        for (int index = 1; index < samples.Length; index++)
        {
            if (samples[index] > peak)
                peak = samples[index];
            if (samples[index] < lowest)
                lowest = samples[index];
        }
        if (peak <= 0 || peak - lowest < 1e-4)
            return 0;
        int best = 2;
        double bestDist = 99;
        bool found = false;
        for (int index = 0; index < samples.Length; index++)
        {
            if (samples[index] < peak - 1e-5)
                continue;
            double dist = Math.Abs(index - 2);
            if (!found || dist < bestDist || (dist == bestDist && index < best))
            {
                found = true;
                best = index;
                bestDist = dist;
            }
        }
        if (best == 0 || best == samples.Length - 1)
            return best - 2;
        double center = samples[best];
        if (Math.Min(center - samples[best - 1], center - samples[best + 1]) < 2e-3)
            return best - 2;
        double bestPos = best;
        double bestValue = -1e9;
        for (double position = best - 1; position <= best + 1.0 + 1e-9; position += 0.01)
        {
            double value = 0;
            for (int index = 0; index < samples.Length; index++)
                value += KeysWeight(position - index) * samples[index];
            if (value > bestValue + 1e-8)
            {
                bestValue = value;
                bestPos = position;
            }
        }
        return bestPos - 2.0;
    }

    /// <summary>沿灰階梯度修列和行。角度不動。候選的分數變差就退回半步，十次都沒有更好就停。</summary>
    static NccHit LucasKanade(NccHit hit, NccLevel field, bool absolute)
    {
        NccPoint[] points = field.Points;
        int step = Math.Max(1, (int)Math.Ceiling(points.Length / 500.0));
        int count = 0;
        for (int i = 0; i < points.Length; i += step)
            count++;
        var offsetCol = new float[count];
        var offsetRow = new float[count];
        var template = new float[count];
        double cosine = Math.Cos(hit.Angle);
        double sine = Math.Sin(hit.Angle);
        int written = 0;
        for (int i = 0; i < points.Length; i += step)
        {
            offsetCol[written] = (float)(points[i].X * cosine + points[i].Y * sine);
            offsetRow[written] = (float)(-points[i].X * sine + points[i].Y * cosine);
            template[written] = points[i].Value;
            written++;
        }
        RotateNcc(points, hit.Angle, out float[] allX, out float[] allY, out float[] allValue);
        double row = hit.Row;
        double col = hit.Column;
        double best = hit.Score;
        var image = new double[count];
        var slopeCol = new double[count];
        var slopeRow = new double[count];
        var reference = new double[count];
        for (int turn = 0; turn < 10; turn++)
        {
            int kept = 0;
            for (int i = 0; i < count; i++)
            {
                double x = col + offsetCol[i];
                double y = row + offsetRow[i];
                if (!Sample(field.Gray, field.Width, field.Height, x, y, out float center))
                    continue;
                if (!Sample(field.Gray, field.Width, field.Height, x + 0.5, y, out float right))
                    continue;
                if (!Sample(field.Gray, field.Width, field.Height, x - 0.5, y, out float left))
                    continue;
                if (!Sample(field.Gray, field.Width, field.Height, x, y + 0.5, out float down))
                    continue;
                if (!Sample(field.Gray, field.Width, field.Height, x, y - 0.5, out float up))
                    continue;
                reference[kept] = template[i];
                image[kept] = center;
                slopeCol[kept] = right - left;
                slopeRow[kept] = down - up;
                kept++;
            }
            if (kept < 16)
                break;
            double meanRef = 0, meanImage = 0, meanCol = 0, meanRow = 0;
            for (int i = 0; i < kept; i++)
            {
                meanRef += reference[i];
                meanImage += image[i];
                meanCol += slopeCol[i];
                meanRow += slopeRow[i];
            }
            meanRef /= kept;
            meanImage /= kept;
            meanCol /= kept;
            meanRow /= kept;
            double a00 = 0, a01 = 0, a11 = 0, b0 = 0, b1 = 0;
            for (int i = 0; i < kept; i++)
            {
                double residual = (reference[i] - meanRef) - (image[i] - meanImage);
                double jacCol = slopeCol[i] - meanCol;
                double jacRow = slopeRow[i] - meanRow;
                a00 += jacCol * jacCol;
                a01 += jacCol * jacRow;
                a11 += jacRow * jacRow;
                b0 += jacCol * residual;
                b1 += jacRow * residual;
            }
            if (!SolveTranslation(a00, a01, a11, b0, b1, out double dCol, out double dRow))
                break;
            bool accepted = false;
            double nextRow = row;
            double nextCol = col;
            for (double scale = 1; scale >= 0.1 - 1e-12; scale *= 0.5)
            {
                double stepCol = dCol * scale;
                double stepRow = dRow * scale;
                if (stepCol < -0.6) stepCol = -0.6;
                if (stepCol > 0.6) stepCol = 0.6;
                if (stepRow < -0.6) stepRow = -0.6;
                if (stepRow > 0.6) stepRow = 0.6;
                double candCol = col + stepCol;
                double candRow = row + stepRow;
                double cand = Correlate(field.Gray, field.Width, field.Height, allX, allY, allValue, candCol, candRow, absolute);
                if (cand + 1e-6 >= best)
                {
                    nextRow = candRow;
                    nextCol = candCol;
                    best = cand;
                    accepted = true;
                    break;
                }
            }
            if (!accepted || Math.Abs(nextRow - row) + Math.Abs(nextCol - col) < 1e-3)
                break;
            row = nextRow;
            col = nextCol;
        }
        if (Math.Abs(row - hit.Row) + Math.Abs(col - hit.Column) <= 1e-3)
            return hit;
        return new NccHit(row, col, hit.Angle, best);
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
