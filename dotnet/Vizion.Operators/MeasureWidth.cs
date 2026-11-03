namespace Vizion;

/// <summary>亮條左右兩側的距離。邊緣落在階躍兩側像素的中間，和次像素邊緣同一種半像素。</summary>
public readonly record struct BarWidth(double Left, double Right, double Width);

public readonly record struct MeasureEdge(double Row, double Column, double Amplitude, double Distance);

public static partial class Operators
{
    public static BarWidth MeasureWidth(ImageBuffer image, int row)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (row < 0 || row >= image.Height || image.Width < 3)
            throw new ArgumentOutOfRangeException(nameof(row));
        byte[] gray = image.GrayBytes();
        int width = image.Width;
        int offset = row * width;
        var line = new byte[width];
        Buffer.BlockCopy(gray, offset, line, 0, width);
        if (NativeCore.TryMeasureRow(line, out double nativeLeft, out double nativeRight, out bool found))
        {
            if (!found)
                throw new InvalidOperationException("這一列看不到亮條的左右兩邊");
            return new BarWidth(nativeLeft, nativeRight, nativeRight - nativeLeft);
        }
        double bestRise = 0;
        double bestFall = 0;
        int riseAt = -1;
        int fallAt = -1;
        for (int column = 0; column < width - 1; column++)
        {
            double delta = gray[offset + column + 1] - gray[offset + column];
            if (delta > bestRise)
            {
                bestRise = delta;
                riseAt = column;
            }
            if (-delta > bestFall)
            {
                bestFall = -delta;
                fallAt = column;
            }
        }
        if (riseAt < 0 || fallAt < 0 || fallAt <= riseAt || bestRise <= 0 || bestFall <= 0)
            throw new InvalidOperationException("這一列看不到亮條的左右兩邊");
        double left = riseAt + 0.5 + EdgeShift(gray, offset, riseAt, width);
        double right = fallAt + 0.5 + EdgeShift(gray, offset, fallAt, width);
        return new BarWidth(left, right, right - left);
    }

    static double EdgeShift(byte[] gray, int offset, int column, int width)
    {
        if (column < 1 || column + 2 >= width)
            return 0;
        double left = Math.Abs(gray[offset + column] - gray[offset + column - 1]);
        double center = Math.Abs(gray[offset + column + 1] - gray[offset + column]);
        double right = Math.Abs(gray[offset + column + 2] - gray[offset + column + 1]);
        double denom = left - 2 * center + right;
        if (Math.Abs(denom) < 1e-6 || center < left || center < right)
            return 0;
        double shift = 0.5 * (left - right) / denom;
        if (shift < -0.5)
            return -0.5;
        if (shift > 0.5)
            return 0.5;
        return shift;
    }

    /// <summary>沿一條掃描線找邊緣。取樣、平滑和拋物線偏差和 Python 的 measure 相同。沒有定義域。</summary>
    public static MeasureEdge[] MeasureScan(
        ImageBuffer image,
        double row1,
        double column1,
        double row2,
        double column2,
        double sigma,
        double threshold,
        string transition)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (sigma < 0 || threshold < 0)
            throw new ArgumentOutOfRangeException(nameof(sigma));
        int kind = ScanTransition(transition);
        byte[] gray = image.GrayBytes();
        if (NativeCore.TryMeasureScan(gray, image.Width, image.Height, column1, row1, column2, row2, sigma, threshold, kind, out MeasureEdge[] native))
            return native;
        return MeasureScanManaged(gray, image.Width, image.Height, column1, row1, column2, row2, sigma, threshold, kind);
    }

    static int ScanTransition(string transition)
    {
        string mode = (transition ?? "all").Trim().Trim('\'').Trim('"').ToLowerInvariant();
        if (mode is "all" or "any" or "全部" or "所有")
            return 0;
        if (mode is "positive" or "pos" or "+" or "dark_to_light" or "由暗到亮")
            return 1;
        if (mode is "negative" or "neg" or "-" or "light_to_dark" or "由亮到暗")
            return 2;
        throw new ArgumentException("Transition 只能是 all、positive 或 negative");
    }

    static bool Finite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

    static int RoundEven(double value)
    {
        double baseValue = Math.Floor(value);
        double diff = value - baseValue;
        int whole = (int)baseValue;
        if (diff > 0.5)
            return whole + 1;
        if (diff < 0.5)
            return whole;
        return (whole & 1) != 0 ? whole + 1 : whole;
    }

    static double SampleGray(byte[] gray, int width, int height, double x, double y)
    {
        if (x < 0 || y < 0 || x > width - 1 || y > height - 1)
            return double.NaN;
        int x0 = (int)Math.Floor(x);
        int y0 = (int)Math.Floor(y);
        int x1 = Math.Min(x0 + 1, width - 1);
        int y1 = Math.Min(y0 + 1, height - 1);
        double dx = x - x0;
        double dy = y - y0;
        double top = gray[y0 * width + x0] * (1 - dx) + gray[y0 * width + x1] * dx;
        double bottom = gray[y1 * width + x0] * (1 - dx) + gray[y1 * width + x1] * dx;
        return top * (1 - dy) + bottom * dy;
    }

    static void SmoothProfile(double[] values, double sigmaPx, double step)
    {
        if (sigmaPx <= 0.05 || values.Length < 5)
            return;
        double sigma = Math.Max(sigmaPx / step, 0.25);
        int radius = RoundEven(sigma * 3.0);
        if (radius < 1)
            radius = 1;
        if (radius > (values.Length - 1) / 2)
            radius = (values.Length - 1) / 2;
        int kernelLength = radius * 2 + 1;
        var kernel = new double[kernelLength];
        double sum = 0;
        for (int offset = -radius; offset <= radius; offset++)
        {
            double tap = Math.Exp(-0.5 * (offset / sigma) * (offset / sigma));
            kernel[offset + radius] = tap;
            sum += tap;
        }
        for (int i = 0; i < kernelLength; i++)
            kernel[i] /= sum;
        var number = new double[values.Length];
        var weight = new double[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            bool finite = Finite(values[i]);
            double filled = finite ? values[i] : 0;
            double mask = finite ? 1 : 0;
            for (int offset = -radius; offset <= radius; offset++)
            {
                int at = i + offset;
                if (at < 0 || at >= values.Length)
                    continue;
                number[at] += filled * kernel[offset + radius];
                weight[at] += mask * kernel[offset + radius];
            }
        }
        for (int i = 0; i < values.Length; i++)
            values[i] = weight[i] > 0.5 ? number[i] / weight[i] : double.NaN;
    }

    static MeasureEdge[] MeasureScanManaged(byte[] gray, int width, int height, double column1, double row1, double column2, double row2, double sigma, double threshold, int transition)
    {
        double dx = column2 - column1;
        double dy = row2 - row1;
        double length = Math.Sqrt(dx * dx + dy * dy);
        if (length < 1.5)
            return Array.Empty<MeasureEdge>();
        const double step = 0.5;
        int count = (int)(length / step) + 1;
        if (count < 5)
            return Array.Empty<MeasureEdge>();
        dx /= length;
        dy /= length;
        var distance = new double[count];
        var xs = new double[count];
        var ys = new double[count];
        var values = new double[count];
        var gradient = new double[count];
        for (int i = 0; i < count; i++)
        {
            distance[i] = length * i / (count - 1);
            xs[i] = column1 + dx * distance[i];
            ys[i] = row1 + dy * distance[i];
            values[i] = SampleGray(gray, width, height, xs[i], ys[i]);
            gradient[i] = double.NaN;
        }
        double spacing = distance[1] - distance[0];
        SmoothProfile(values, sigma, step);
        for (int i = 1; i < count - 1; i++)
        {
            if (Finite(values[i - 1]) && Finite(values[i + 1]))
                gradient[i] = (values[i + 1] - values[i - 1]) / (2 * step);
        }
        var hits = new List<MeasureEdge>();
        for (int index = 2; index < count - 2; index++)
        {
            double left = gradient[index - 1];
            double mid = gradient[index];
            double right = gradient[index + 1];
            if (!Finite(left) || !Finite(mid) || !Finite(right))
                continue;
            bool positive = mid >= threshold && mid >= left && mid > right;
            bool negative = mid <= -threshold && mid <= left && mid < right;
            if (transition == 1 && !positive)
                continue;
            if (transition == 2 && !negative)
                continue;
            if (transition == 0 && !(positive || negative))
                continue;
            double denom = left - 2 * mid + right;
            double delta = Math.Abs(denom) < 1e-9 ? 0 : 0.5 * (left - right) / denom;
            if (delta < -1)
                delta = -1;
            if (delta > 1)
                delta = 1;
            double at = index + delta;
            int i0 = (int)Math.Floor(at);
            int i1 = Math.Min(i0 + 1, count - 1);
            double frac = at - i0;
            hits.Add(new MeasureEdge(
                ys[i0] * (1 - frac) + ys[i1] * frac,
                xs[i0] * (1 - frac) + xs[i1] * frac,
                mid,
                at * spacing));
        }
        return hits.ToArray();
    }
}
