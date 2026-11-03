namespace Vizion;

/// <summary>亮條左右兩側的距離。邊緣落在階躍兩側像素的中間，和次像素邊緣同一種半像素。</summary>
public readonly record struct BarWidth(double Left, double Right, double Width);

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
}
