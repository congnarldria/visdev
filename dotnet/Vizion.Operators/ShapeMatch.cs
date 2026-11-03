namespace Vizion;

/// <summary>
/// 簡化的形狀比對。教導框中心是原點，和 teach_shape_model 相同。
/// 這不是 visdev_match.pyd：沒有金字塔，也沒有 least_squares。分數是邊緣方向有多一致，不是 pyd 的 0.99。
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
        Sobel(scene, out float[] sgx, out float[] sgy, out float[] smag);
        var hits = new List<ShapeHit>();
        const double step = 4.0 * Math.PI / 180.0;
        for (double angle = -Math.PI; angle < Math.PI - 1e-6; angle += step)
        {
            Rotate(model, angle, out float[] rx, out float[] ry, out float[] rgx, out float[] rgy);
            for (int row = 8; row < scene.Height - 8; row += 4)
            {
                for (int col = 8; col < scene.Width - 8; col += 4)
                {
                    double score = Score(sgx, sgy, smag, scene.Width, scene.Height, col, row, rx, ry, rgx, rgy, minContrast);
                    if (score >= minScore)
                        hits.Add(new ShapeHit(row, col, angle, score));
                }
            }
        }
        var kept = new List<ShapeHit>();
        foreach (var hit in hits.OrderByDescending(item => item.Score))
        {
            bool near = false;
            foreach (var other in kept)
            {
                double dRow = other.Row - hit.Row;
                double dCol = other.Column - hit.Column;
                if (dRow * dRow + dCol * dCol < 70 * 70)
                {
                    near = true;
                    break;
                }
            }
            if (!near)
                kept.Add(Refine(hit, model, sgx, sgy, smag, scene.Width, scene.Height, minContrast));
            if (kept.Count == 8)
                break;
        }
        return kept.ToArray();
    }

    readonly record struct ModelPoint(float X, float Y, float Gx, float Gy);

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

    static ShapeHit Refine(ShapeHit hit, ModelPoint[] model, float[] gx, float[] gy, float[] mag, int width, int height, double minContrast)
    {
        var best = hit;
        for (int pass = 0; pass < 2; pass++)
        {
            double angleStep = pass == 0 ? 2.0 * Math.PI / 180.0 : 0.5 * Math.PI / 180.0;
            int span = pass == 0 ? 4 : 2;
            for (int dRow = -span; dRow <= span; dRow++)
            {
                for (int dCol = -span; dCol <= span; dCol++)
                {
                    for (int dAngle = -2; dAngle <= 2; dAngle++)
                    {
                        double row = hit.Row + dRow * (pass == 0 ? 1 : 1);
                        double col = hit.Column + dCol;
                        double angle = hit.Angle + dAngle * angleStep;
                        Rotate(model, angle, out float[] rx, out float[] ry, out float[] rgx, out float[] rgy);
                        double score = Score(gx, gy, mag, width, height, col, row, rx, ry, rgx, rgy, minContrast);
                        if (score > best.Score)
                            best = new ShapeHit(row, col, angle, score);
                    }
                }
            }
            hit = best;
        }
        return best;
    }

    static double Score(
        float[] gx, float[] gy, float[] mag, int width, int height,
        double column, double row,
        float[] mx, float[] my, float[] mgx, float[] mgy, double minContrast)
    {
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
            float sceneGy = (1 - wy) * ((1 - wx) * gy[i00] + wy * gy[i10]) + wy * ((1 - wx) * gy[i01] + wy * gy[i11]);
            float inv = 1f / Math.Max(sceneMag, 1e-3f);
            sum += Math.Max(0, mgx[i] * sceneGx * inv + mgy[i] * sceneGy * inv);
        }
        return sum / count;
    }

    static void Sobel(ImageBuffer image, out float[] gx, out float[] gy, out float[] mag)
    {
        int width = image.Width;
        int height = image.Height;
        byte[] gray = image.GrayBytes();
        gx = new float[width * height];
        gy = new float[width * height];
        mag = new float[width * height];
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
