namespace Vizion;

/// <summary>
/// 簡化的形狀比對。教導框中心是原點，和 teach_shape_model 相同。
/// 先在高斯金字塔的粗圖上每 4 度搜尋，再一層層回到原圖。
/// 原圖上用分數曲面的拋物線修列、行和角度，再沿邊緣法線做最小二乘。
/// 這不是 visdev_match.pyd。分數是邊緣方向有多一致，不是 pyd 的 0.99。
/// </summary>
public readonly record struct ShapeHit(double Row, double Column, double Angle, double Score);

public static partial class Operators
{
    public static ShapeHit[] FindShape(
        ImageBuffer scene,
        ImageBuffer template,
        int row1,
        int column1,
        int row2,
        int column2,
        double contrast = 40,
        double minContrast = 10,
        double minScore = 0.55)
    {
        if (scene == null || template == null)
            throw new ArgumentNullException(nameof(scene));
        if (contrast < 0 || minContrast < 0)
            throw new ArgumentOutOfRangeException(nameof(contrast));
        var model = BuildModel(template, row1, column1, row2, column2, contrast);
        if (model.Length < 12)
            throw new InvalidOperationException("模板邊緣點太少。請降低 Contrast，或確認模板有清楚輪廓。");

        var levels = BuildFields(scene, 3);
        int coarseIndex = levels.Count - 1;
        var coarse = levels[coarseIndex];
        var hits = new List<ShapeHit>();
        const double angleStep = 4.0 * Math.PI / 180.0;
        int stride = 2;
        int margin = 3;
        double coarseMin = minScore * 0.62;
        for (double angle = -Math.PI; angle < Math.PI - 1e-6; angle += angleStep)
        {
            Prepare(model, angle, coarse.Scale, out float[] rx, out float[] ry, out float[] rgx, out float[] rgy);
            for (int row = margin; row < coarse.Height - margin; row += stride)
            {
                for (int col = margin; col < coarse.Width - margin; col += stride)
                {
                    double score = Score(coarse.Gx, coarse.Gy, coarse.Mag, coarse.Width, coarse.Height, col, row, rx, ry, rgx, rgy, minContrast);
                    if (score >= coarseMin)
                        hits.Add(new ShapeHit(row * coarse.Scale, col * coarse.Scale, angle, score));
                }
            }
        }

        var kept = Suppress(hits, 70, 16);
        for (int index = coarseIndex - 1; index >= 0; index--)
        {
            var field = levels[index];
            double angleWindow = index == 0 ? 2.0 * Math.PI / 180.0 : angleStep;
            var promoted = new List<ShapeHit>();
            foreach (var hit in kept)
            {
                var best = Promote(hit, model, field, angleWindow, minContrast);
                if (best.Score >= minScore * 0.75)
                    promoted.Add(best);
            }
            kept = Suppress(promoted, 70, 8);
        }

        var refined = new List<ShapeHit>();
        var full = levels[0];
        foreach (var hit in kept)
        {
            if (hit.Score < minScore)
                continue;
            var posed = SnapHalfPixel(hit, model, full, minContrast);
            posed = PolishAngle(posed, model, full, minContrast);
            posed = SnapHalfPixel(posed, model, full, minContrast);
            posed = Parabolic(posed, model, full, minContrast);
            var fitted = LeastSquares(posed, model, full, minContrast);
            if (fitted.Score + 1e-4 < posed.Score)
                fitted = posed;
            fitted = fitted with { Angle = WrapAngle(fitted.Angle) };
            if (fitted.Score >= minScore * 0.9)
                refined.Add(fitted);
        }
        return Suppress(refined, 70, 8).ToArray();
    }

    readonly record struct ModelPoint(float X, float Y, float Gx, float Gy);

    readonly record struct Field(int Width, int Height, int Scale, float[] Gx, float[] Gy, float[] Mag);

    static List<Field> BuildFields(ImageBuffer scene, int maxLevels)
    {
        var levels = new List<Field>();
        byte[] gray = scene.GrayBytes();
        int width = scene.Width;
        int height = scene.Height;
        int scale = 1;
        for (int level = 0; level < maxLevels; level++)
        {
            SobelBytes(gray, width, height, out float[] gx, out float[] gy, out float[] mag);
            levels.Add(new Field(width, height, scale, gx, gy, mag));
            if (level + 1 == maxLevels || width < 48 || height < 48)
                break;
            PyrDown(gray, width, height, out gray, out width, out height);
            scale *= 2;
        }
        return levels;
    }

    /// <summary>5 點高斯 [1,4,6,4,1]，再隔一點取樣。偶數邊長時，粗圖一點對齊細圖的偶數座標。</summary>
    static void PyrDown(byte[] source, int width, int height, out byte[] destination, out int destWidth, out int destHeight)
    {
        destWidth = (width + 1) / 2;
        destHeight = (height + 1) / 2;
        int[] kernel = { 1, 4, 6, 4, 1 };
        var horizontal = new int[width * height];
        for (int row = 0; row < height; row++)
        {
            int rowOffset = row * width;
            for (int col = 0; col < width; col++)
            {
                int sum = 0;
                for (int tap = -2; tap <= 2; tap++)
                    sum += kernel[tap + 2] * source[rowOffset + Reflect(col + tap, width)];
                horizontal[rowOffset + col] = sum;
            }
        }
        destination = new byte[destWidth * destHeight];
        for (int row = 0; row < destHeight; row++)
        {
            int sourceRow = row * 2;
            for (int col = 0; col < destWidth; col++)
            {
                int sourceCol = col * 2;
                int sum = 0;
                for (int tap = -2; tap <= 2; tap++)
                {
                    int rr = Reflect(sourceRow + tap, height);
                    sum += kernel[tap + 2] * horizontal[rr * width + sourceCol];
                }
                int value = (sum + 128) / 256;
                if (value < 0)
                    value = 0;
                if (value > 255)
                    value = 255;
                destination[row * destWidth + col] = (byte)value;
            }
        }
    }

    static ShapeHit Promote(ShapeHit hit, ModelPoint[] model, Field field, double angleWindow, double minContrast)
    {
        var best = new ShapeHit(hit.Row, hit.Column, hit.Angle, -1);
        double baseRow = hit.Row / field.Scale;
        double baseCol = hit.Column / field.Scale;
        for (int angleStep = -2; angleStep <= 2; angleStep++)
        {
            double angle = hit.Angle + angleStep * (angleWindow / 2.0);
            Prepare(model, angle, field.Scale, out float[] rx, out float[] ry, out float[] rgx, out float[] rgy);
            for (int dRow = -2; dRow <= 2; dRow++)
            {
                for (int dCol = -2; dCol <= 2; dCol++)
                {
                    double row = baseRow + dRow;
                    double col = baseCol + dCol;
                    if (row < 2 || col < 2 || row >= field.Height - 2 || col >= field.Width - 2)
                        continue;
                    double score = Score(field.Gx, field.Gy, field.Mag, field.Width, field.Height, col, row, rx, ry, rgx, rgy, minContrast);
                    if (score > best.Score)
                        best = new ShapeHit(row * field.Scale, col * field.Scale, angle, score);
                }
            }
        }
        return best.Score < 0 ? hit : best;
    }

    static ShapeHit Parabolic(ShapeHit hit, ModelPoint[] model, Field field, double minContrast)
    {
        double row = hit.Row;
        double col = hit.Column;
        double angle = hit.Angle;
        double step = 0.8 * Math.PI / 180.0;
        const double spacing = 0.5;
        for (int pass = 0; pass < 2; pass++)
        {
            double center = ScorePose(model, field, col, row, angle, minContrast);
            double dRow = Parabola(
                ScorePose(model, field, col, row - spacing, angle, minContrast),
                center,
                ScorePose(model, field, col, row + spacing, angle, minContrast));
            double dCol = Parabola(
                ScorePose(model, field, col - spacing, row, angle, minContrast),
                center,
                ScorePose(model, field, col + spacing, row, angle, minContrast));
            double dAngle = Parabola(
                ScorePose(model, field, col, row, angle - step, minContrast),
                center,
                ScorePose(model, field, col, row, angle + step, minContrast));
            row += dRow * spacing;
            col += dCol * spacing;
            angle += dAngle * step;
            step *= 0.5;
        }
        double score = ScorePose(model, field, col, row, angle, minContrast);
        if (score + 1e-4 < hit.Score)
            return hit;
        return new ShapeHit(row, col, angle, score);
    }

    /// <summary>整數峰旁邊半像素若分數更高，先移過去。方塊這類平頂才不會停在格子角落。</summary>
    static ShapeHit SnapHalfPixel(ShapeHit hit, ModelPoint[] model, Field field, double minContrast)
    {
        var best = hit;
        int row0 = (int)Math.Round(hit.Row);
        int col0 = (int)Math.Round(hit.Column);
        for (int dRow = -2; dRow <= 2; dRow++)
        {
            for (int dCol = -2; dCol <= 2; dCol++)
            {
                double row = row0 + dRow * 0.5;
                double col = col0 + dCol * 0.5;
                if (row < 2 || col < 2 || row >= field.Height - 2 || col >= field.Width - 2)
                    continue;
                double score = ScorePose(model, field, col, row, hit.Angle, minContrast);
                if (score > best.Score)
                    best = new ShapeHit(row, col, hit.Angle, score);
            }
        }
        return best;
    }

    static ShapeHit PolishAngle(ShapeHit hit, ModelPoint[] model, Field field, double minContrast)
    {
        var best = hit;
        double step = 0.5 * Math.PI / 180.0;
        for (int index = -6; index <= 6; index++)
        {
            double angle = hit.Angle + index * step;
            double score = ScorePose(model, field, hit.Column, hit.Row, angle, minContrast);
            if (score > best.Score)
                best = new ShapeHit(hit.Row, hit.Column, angle, score);
        }
        return best;
    }

    static double WrapAngle(double angle)
    {
        while (angle <= -Math.PI)
            angle += 2 * Math.PI;
        while (angle > Math.PI)
            angle -= 2 * Math.PI;
        return angle;
    }

    static double Parabola(double left, double center, double right)
    {
        double denom = left - 2 * center + right;
        if (Math.Abs(denom) < 1e-6 || center < left || center < right)
            return 0;
        double shift = 0.5 * (left - right) / denom;
        if (shift < -0.75)
            return -0.75;
        if (shift > 0.75)
            return 0.75;
        return shift;
    }

    static ShapeHit LeastSquares(ShapeHit hit, ModelPoint[] model, Field field, double minContrast)
    {
        int step = Math.Max(1, (int)Math.Ceiling(model.Length / 80.0));
        double row = hit.Row;
        double col = hit.Column;
        double angle = hit.Angle;
        double angleLimit = 1.5 * Math.PI / 180.0;
        for (int turn = 0; turn < 4; turn++)
        {
            double cosine = Math.Cos(angle);
            double sine = Math.Sin(angle);
            double a00 = 0, a01 = 0, a02 = 0, a11 = 0, a12 = 0, a22 = 0;
            double b0 = 0, b1 = 0, b2 = 0;
            int kept = 0;
            for (int i = 0; i < model.Length; i += step)
            {
                double rotatedCol = model[i].X * cosine + model[i].Y * sine;
                double rotatedRow = -model[i].X * sine + model[i].Y * cosine;
                double normalCol = model[i].Gx * cosine + model[i].Gy * sine;
                double normalRow = -model[i].Gx * sine + model[i].Gy * cosine;
                double sampleCol = col + rotatedCol;
                double sampleRow = row + rotatedRow;
                if (!Sample(field.Mag, field.Width, field.Height, sampleCol, sampleRow, out float center))
                    continue;
                if (!Sample(field.Mag, field.Width, field.Height, sampleCol - normalCol, sampleRow - normalRow, out float before))
                    continue;
                if (!Sample(field.Mag, field.Width, field.Height, sampleCol + normalCol, sampleRow + normalRow, out float after))
                    continue;
                double denom = before - 2.0 * center + after;
                if (center < before || center < after || Math.Abs(denom) <= 1e-3 || center < minContrast)
                    continue;
                double offset = 0.5 * (before - after) / denom;
                if (offset < -0.6)
                    offset = -0.6;
                if (offset > 0.6)
                    offset = 0.6;
                double twist = normalCol * rotatedRow - normalRow * rotatedCol;
                a00 += normalCol * normalCol;
                a01 += normalCol * normalRow;
                a02 += normalCol * twist;
                a11 += normalRow * normalRow;
                a12 += normalRow * twist;
                a22 += twist * twist;
                b0 += normalCol * offset;
                b1 += normalRow * offset;
                b2 += twist * offset;
                kept++;
            }
            if (kept < 8)
                break;
            if (!SolveSymmetric3(a00, a01, a02, a11, a12, a22, b0, b1, b2, out double dCol, out double dRow, out double dAngle)
                && !SolveTranslation(a00, a01, a11, b0, b1, out dCol, out dRow))
                break;
            if (dCol < -0.6) dCol = -0.6;
            if (dCol > 0.6) dCol = 0.6;
            if (dRow < -0.6) dRow = -0.6;
            if (dRow > 0.6) dRow = 0.6;
            if (dAngle < -angleLimit) dAngle = -angleLimit;
            if (dAngle > angleLimit) dAngle = angleLimit;
            if (Math.Abs(dCol) + Math.Abs(dRow) + Math.Abs(dAngle) < 1e-4)
                break;
            col += dCol;
            row += dRow;
            angle += dAngle;
        }
        double score = ScorePose(model, field, col, row, angle, minContrast);
        return new ShapeHit(row, col, angle, score);
    }

    static bool SolveTranslation(double a00, double a01, double a11, double b0, double b1, out double dCol, out double dRow)
    {
        double det = a00 * a11 - a01 * a01;
        if (Math.Abs(det) > 1e-8)
        {
            dCol = (a11 * b0 - a01 * b1) / det;
            dRow = (a00 * b1 - a01 * b0) / det;
            return true;
        }
        dCol = 0;
        dRow = 0;
        if (a00 > 1e-8)
        {
            dCol = b0 / a00;
            return true;
        }
        if (a11 > 1e-8)
        {
            dRow = b1 / a11;
            return true;
        }
        return false;
    }

    static bool SolveSymmetric3(
        double a00, double a01, double a02,
        double a11, double a12, double a22,
        double b0, double b1, double b2,
        out double x0, out double x1, out double x2)
    {
        double[,] rows =
        {
            { a00, a01, a02, b0 },
            { a01, a11, a12, b1 },
            { a02, a12, a22, b2 },
        };
        for (int col = 0; col < 3; col++)
        {
            int pivot = col;
            double best = Math.Abs(rows[col, col]);
            for (int row = col + 1; row < 3; row++)
            {
                double value = Math.Abs(rows[row, col]);
                if (value > best)
                {
                    best = value;
                    pivot = row;
                }
            }
            if (best < 1e-12)
            {
                x0 = x1 = x2 = 0;
                return false;
            }
            if (pivot != col)
            {
                for (int k = col; k < 4; k++)
                {
                    double swap = rows[col, k];
                    rows[col, k] = rows[pivot, k];
                    rows[pivot, k] = swap;
                }
            }
            double div = rows[col, col];
            for (int k = col; k < 4; k++)
                rows[col, k] /= div;
            for (int row = 0; row < 3; row++)
            {
                if (row == col)
                    continue;
                double factor = rows[row, col];
                if (factor == 0)
                    continue;
                for (int k = col; k < 4; k++)
                    rows[row, k] -= factor * rows[col, k];
            }
        }
        x0 = rows[0, 3];
        x1 = rows[1, 3];
        x2 = rows[2, 3];
        return true;
    }

    static bool Sample(float[] image, int width, int height, double x, double y, out float value)
    {
        int x0 = (int)Math.Floor(x);
        int y0 = (int)Math.Floor(y);
        if (x0 < 0 || y0 < 0 || x0 + 1 >= width || y0 + 1 >= height)
        {
            value = 0;
            return false;
        }
        float wx = (float)(x - x0);
        float wy = (float)(y - y0);
        int i00 = y0 * width + x0;
        float top = (1 - wx) * image[i00] + wx * image[i00 + 1];
        float bottom = (1 - wx) * image[i00 + width] + wx * image[i00 + width + 1];
        value = (1 - wy) * top + wy * bottom;
        return true;
    }

    static double ScorePose(ModelPoint[] model, Field field, double column, double row, double angle, double minContrast)
    {
        Prepare(model, angle, field.Scale, out float[] rx, out float[] ry, out float[] rgx, out float[] rgy);
        return Score(field.Gx, field.Gy, field.Mag, field.Width, field.Height, column / field.Scale, row / field.Scale, rx, ry, rgx, rgy, minContrast);
    }

    static void Prepare(ModelPoint[] model, double angle, double scale, out float[] x, out float[] y, out float[] gx, out float[] gy)
    {
        Rotate(model, angle, out x, out y, out gx, out gy);
        float inv = (float)(1.0 / scale);
        if (inv == 1f)
            return;
        for (int i = 0; i < x.Length; i++)
        {
            x[i] *= inv;
            y[i] *= inv;
        }
    }

    static List<ShapeHit> Suppress(List<ShapeHit> hits, double minDistance, int maximum)
    {
        var kept = new List<ShapeHit>();
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

    static ModelPoint[] BuildModel(ImageBuffer template, int row1, int column1, int row2, int column2, double contrast)
    {
        Sobel(template, out float[] gx, out float[] gy, out float[] mag);
        int width = template.Width;
        int height = template.Height;
        double originRow = (row1 + row2) / 2.0;
        double originCol = (column1 + column2) / 2.0;
        var points = new List<ModelPoint>();
        for (int row = Math.Max(1, row1); row <= Math.Min(height - 2, row2); row++)
        {
            for (int col = Math.Max(1, column1); col <= Math.Min(width - 2, column2); col++)
            {
                int index = row * width + col;
                if (mag[index] < contrast || !IsRidge(mag, gx, gy, width, height, col, row))
                    continue;
                float scale = Math.Max(mag[index], 1e-3f);
                points.Add(new ModelPoint((float)(col - originCol), (float)(row - originRow), gx[index] / scale, gy[index] / scale));
            }
        }
        if (points.Count <= 160)
            return points.ToArray();
        int stride = (int)Math.Ceiling(points.Count / 160.0);
        var kept = new List<ModelPoint>();
        for (int i = 0; i < points.Count; i += stride)
            kept.Add(points[i]);
        return kept.ToArray();
    }

    static void Rotate(ModelPoint[] model, double angle, out float[] x, out float[] y, out float[] gx, out float[] gy)
    {
        float cosine = (float)Math.Cos(angle);
        float sine = (float)Math.Sin(angle);
        x = new float[model.Length];
        y = new float[model.Length];
        gx = new float[model.Length];
        gy = new float[model.Length];
        for (int i = 0; i < model.Length; i++)
        {
            x[i] = model[i].X * cosine + model[i].Y * sine;
            y[i] = -model[i].X * sine + model[i].Y * cosine;
            gx[i] = model[i].Gx * cosine + model[i].Gy * sine;
            gy[i] = -model[i].Gx * sine + model[i].Gy * cosine;
        }
    }

    static double Score(
        float[] gx, float[] gy, float[] mag, int width, int height,
        double column, double row,
        float[] mx, float[] my, float[] mgx, float[] mgy, double minContrast)
    {
        if (NativeCore.TryShapeScore(gx, gy, mag, width, height, column, row, mx, my, mgx, mgy, minContrast, out double nativeScore))
            return nativeScore;
        double sum = 0;
        int count = mx.Length;
        for (int i = 0; i < count; i++)
        {
            double x = column + mx[i];
            double y = row + my[i];
            if (x < 1 || y < 1 || x >= width - 2 || y >= height - 2)
                continue;
            int x0 = (int)Math.Floor(x);
            int y0 = (int)Math.Floor(y);
            float wx = (float)(x - x0);
            float wy = (float)(y - y0);
            int i00 = y0 * width + x0;
            int i10 = i00 + 1;
            int i01 = i00 + width;
            int i11 = i01 + 1;
            float sceneMag = (1 - wy) * ((1 - wx) * mag[i00] + wx * mag[i10]) + wy * ((1 - wx) * mag[i01] + wx * mag[i11]);
            if (sceneMag < minContrast)
                continue;
            float sceneGx = (1 - wy) * ((1 - wx) * gx[i00] + wx * gx[i10]) + wy * ((1 - wx) * gx[i01] + wx * gx[i11]);
            float sceneGy = (1 - wy) * ((1 - wx) * gy[i00] + wx * gy[i10]) + wy * ((1 - wx) * gy[i01] + wx * gy[i11]);
            float inv = 1f / Math.Max(sceneMag, 1e-3f);
            sum += Math.Max(0, mgx[i] * sceneGx * inv + mgy[i] * sceneGy * inv);
        }
        return sum / count;
    }

    static void Sobel(ImageBuffer image, out float[] gx, out float[] gy, out float[] mag) =>
        SobelBytes(image.GrayBytes(), image.Width, image.Height, out gx, out gy, out mag);

    static void SobelBytes(byte[] gray, int width, int height, out float[] gx, out float[] gy, out float[] mag)
    {
        gx = new float[width * height];
        gy = new float[width * height];
        mag = new float[width * height];
        if (NativeCore.TrySobel(gray, width, height, gx, gy, mag))
            return;
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                float At(int y, int x) => gray[Reflect(y, height) * width + Reflect(x, width)];
                float dx =
                    -At(row - 1, col - 1) + At(row - 1, col + 1)
                    - 2 * At(row, col - 1) + 2 * At(row, col + 1)
                    - At(row + 1, col - 1) + At(row + 1, col + 1);
                float dy =
                    -At(row - 1, col - 1) - 2 * At(row - 1, col) - At(row - 1, col + 1)
                    + At(row + 1, col - 1) + 2 * At(row + 1, col) + At(row + 1, col + 1);
                int index = row * width + col;
                gx[index] = dx;
                gy[index] = dy;
                mag[index] = (float)Math.Sqrt(dx * dx + dy * dy);
            }
        }
    }

    static int Reflect(int position, int length)
    {
        if ((uint)position < (uint)length)
            return position;
        if (length == 1)
            return 0;
        do
        {
            if (position < 0)
                position = -position - 1;
            else
                position = length - 1 - (position - length);
        }
        while ((uint)position >= (uint)length);
        return position;
    }

    static bool IsRidge(float[] mag, float[] gx, float[] gy, int width, int height, int col, int row)
    {
        int index = row * width + col;
        float center = mag[index];
        double angle = (Math.Atan2(gy[index], gx[index]) * 180.0 / Math.PI + 180.0) % 180.0;
        float At(int y, int x)
        {
            if (y < 0 || x < 0 || y >= height || x >= width)
                return 0;
            return mag[y * width + x];
        }
        if (angle < 22.5 || angle >= 157.5)
            return center >= At(row, col - 1) && center >= At(row, col + 1);
        if (angle < 67.5)
            return center >= At(row - 1, col - 1) && center >= At(row + 1, col + 1);
        if (angle < 112.5)
            return center >= At(row - 1, col) && center >= At(row + 1, col);
        return center >= At(row - 1, col + 1) && center >= At(row + 1, col - 1);
    }
}
