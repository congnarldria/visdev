namespace Vizion;

public static partial class Operators
{
    /// <summary>
    /// 把有 k1 畸變的圖校正成直線模型。輸出像素到原圖的位置是 x' = x (1 + k1 r²)，雙線性取樣，圖外補 0。
    /// 不改 Python 的九個參數。
    /// </summary>
    public static ImageBuffer Undistort(ImageBuffer image, Intrinsics camera)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        byte[] source = image.GrayBytes();
        int width = image.Width;
        int height = image.Height;
        var result = new byte[source.Length];
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                Distort(camera, column, row, out double sourceColumn, out double sourceRow);
                result[row * width + column] = SampleBilinear(source, width, height, sourceColumn, sourceRow);
            }
        }
        return ImageBuffer.Gray(width, height, result);
    }

    /// <summary>
    /// 影像點投到相機前方 Z=0 的平面。camera = R * world + t。Scale 再乘上世界座標，用來換成公釐。
    /// </summary>
    public static (double X, double Y) ImageToPlane(
        double row,
        double column,
        Intrinsics camera,
        double[,] rotation,
        double tx,
        double ty,
        double tz,
        double scale = 1)
    {
        UndistortNormalized(camera, column, row, out double x, out double y);
        double planeX = rotation[0, 2];
        double planeY = rotation[1, 2];
        double planeZ = rotation[2, 2];
        double denom = planeX * x + planeY * y + planeZ;
        if (Math.Abs(denom) < 1e-12)
            throw new InvalidOperationException("這條視線和平面平行");
        double depth = (planeX * tx + planeY * ty + planeZ * tz) / denom;
        double cameraX = depth * x - tx;
        double cameraY = depth * y - ty;
        double cameraZ = depth - tz;
        double worldX = rotation[0, 0] * cameraX + rotation[1, 0] * cameraY + rotation[2, 0] * cameraZ;
        double worldY = rotation[0, 1] * cameraX + rotation[1, 1] * cameraY + rotation[2, 1] * cameraZ;
        return (worldX * scale, worldY * scale);
    }

    /// <summary>
    /// 先去掉 k1，再量這一列亮條的左右邊緣，投到 Z=0 後的距離。單位跟著姿態和 Scale。
    /// </summary>
    public static double MeasurePlaneWidth(
        ImageBuffer image,
        int row,
        Intrinsics camera,
        double[,] rotation,
        double tx,
        double ty,
        double tz,
        double scale = 1)
    {
        var straight = Undistort(image, camera);
        var ideal = new Intrinsics(camera.Fx, camera.Fy, camera.Cx, camera.Cy);
        var width = MeasureWidth(straight, row);
        var left = ImageToPlane(row, width.Left, ideal, rotation, tx, ty, tz, scale);
        var right = ImageToPlane(row, width.Right, ideal, rotation, tx, ty, tz, scale);
        double dx = right.X - left.X;
        double dy = right.Y - left.Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    static void Distort(Intrinsics camera, double column, double row, out double sourceColumn, out double sourceRow)
    {
        double x = (column - camera.Cx) / camera.Fx;
        double y = (row - camera.Cy) / camera.Fy;
        double radius = x * x + y * y;
        double factor = 1 + camera.K1 * radius;
        sourceColumn = camera.Fx * x * factor + camera.Cx;
        sourceRow = camera.Fy * y * factor + camera.Cy;
    }

    static void UndistortNormalized(Intrinsics camera, double column, double row, out double x, out double y)
    {
        x = (column - camera.Cx) / camera.Fx;
        y = (row - camera.Cy) / camera.Fy;
        if (camera.K1 == 0)
            return;
        for (int iteration = 0; iteration < 8; iteration++)
        {
            double radius = x * x + y * y;
            double factor = 1 + camera.K1 * radius;
            if (Math.Abs(factor) < 1e-12)
                break;
            double nextX = ((column - camera.Cx) / camera.Fx) / factor;
            double nextY = ((row - camera.Cy) / camera.Fy) / factor;
            if (Math.Abs(nextX - x) + Math.Abs(nextY - y) < 1e-12)
            {
                x = nextX;
                y = nextY;
                break;
            }
            x = nextX;
            y = nextY;
        }
    }

    static byte SampleBilinear(byte[] source, int width, int height, double column, double row)
    {
        if (column < 0 || row < 0 || column > width - 1 || row > height - 1)
            return 0;
        int x0 = (int)Math.Floor(column);
        int y0 = (int)Math.Floor(row);
        int x1 = Math.Min(width - 1, x0 + 1);
        int y1 = Math.Min(height - 1, y0 + 1);
        double wx = column - x0;
        double wy = row - y0;
        double top = source[y0 * width + x0] * (1 - wx) + source[y0 * width + x1] * wx;
        double bottom = source[y1 * width + x0] * (1 - wx) + source[y1 * width + x1] * wx;
        double value = top * (1 - wy) + bottom * wy;
        int rounded = (int)Math.Round(value, MidpointRounding.AwayFromZero);
        if (rounded < 0)
            return 0;
        if (rounded > 255)
            return 255;
        return (byte)rounded;
    }
}
