namespace Vizion;

public static partial class Operators
{
    /// <summary>
    /// 繞影像中心旋轉。正角度在畫面上順時針，和方向角同一套。最近鄰取樣，轉出圖外的像素補 0。
    /// </summary>
    public static ImageBuffer RotateNearest(ImageBuffer image, double degrees)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        byte[] source = image.GrayBytes();
        int width = image.Width;
        int height = image.Height;
        double centerColumn = (width - 1) / 2.0;
        double centerRow = (height - 1) / 2.0;
        double radians = degrees * Math.PI / 180.0;
        double cosine = Math.Cos(radians);
        double sine = Math.Sin(radians);
        var result = new byte[source.Length];
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                double dx = column - centerColumn;
                double dy = row - centerRow;
                double sourceColumn = centerColumn + dx * cosine - dy * sine;
                double sourceRow = centerRow + dx * sine + dy * cosine;
                int sampleColumn = (int)Math.Round(sourceColumn, MidpointRounding.AwayFromZero);
                int sampleRow = (int)Math.Round(sourceRow, MidpointRounding.AwayFromZero);
                if ((uint)sampleRow >= (uint)height || (uint)sampleColumn >= (uint)width)
                    continue;
                result[row * width + column] = source[sampleRow * width + sampleColumn];
            }
        }
        return ImageBuffer.Gray(width, height, result);
    }

    /// <summary>繞著指定點把列、行轉正。正角度在畫面上順時針。</summary>
    public static (double Row, double Column) RotatePoint(double row, double column, double degrees, double pivotRow, double pivotColumn)
    {
        double radians = degrees * Math.PI / 180.0;
        double cosine = Math.Cos(radians);
        double sine = Math.Sin(radians);
        double dx = column - pivotColumn;
        double dy = row - pivotRow;
        return (pivotRow + dx * sine + dy * cosine, pivotColumn + dx * cosine - dy * sine);
    }
}
