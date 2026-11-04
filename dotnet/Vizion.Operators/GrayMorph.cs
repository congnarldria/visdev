namespace Vizion;

public static partial class Operators
{
    /// <summary>灰階矩形侵蝕。每個像素換成遮罩裡最暗的值。原點在 k/2，偶數不補成奇數。圖外當成 255。</summary>
    public static ImageBuffer GrayErosion(ImageBuffer image, int maskHeight, int maskWidth)
    {
        if (maskHeight < 1 || maskWidth < 1)
            throw new ArgumentOutOfRangeException(nameof(maskHeight), "遮罩至少要 1x1");
        return ApplyRectangle(image, maskHeight, maskWidth, dark: true);
    }

    /// <summary>灰階矩形膨脹。每個像素換成遮罩裡最亮的值。原點在 k/2。圖外當成 0。</summary>
    public static ImageBuffer GrayDilation(ImageBuffer image, int maskHeight, int maskWidth)
    {
        if (maskHeight < 1 || maskWidth < 1)
            throw new ArgumentOutOfRangeException(nameof(maskWidth), "遮罩至少要 1x1");
        return ApplyRectangle(image, maskHeight, maskWidth, dark: false);
    }

    /// <summary>灰階圓形侵蝕。半徑 1 是十字。圖外當成 255。</summary>
    public static ImageBuffer GrayErosionCircle(ImageBuffer image, int radius) =>
        ApplyCircle(image, radius, dark: true);

    /// <summary>灰階圓形膨脹。半徑 1 是十字。圖外當成 0。</summary>
    public static ImageBuffer GrayDilationCircle(ImageBuffer image, int radius) =>
        ApplyCircle(image, radius, dark: false);

    /// <summary>灰階矩形開運算。先侵蝕再膨脹。比遮罩小的亮點會消失。</summary>
    public static ImageBuffer GrayOpenRectangle(ImageBuffer image, int maskHeight, int maskWidth) =>
        GrayDilation(GrayErosion(image, maskHeight, maskWidth), maskHeight, maskWidth);

    /// <summary>灰階矩形閉運算。先膨脹再侵蝕。比遮罩小的暗點會補上。</summary>
    public static ImageBuffer GrayCloseRectangle(ImageBuffer image, int maskHeight, int maskWidth) =>
        GrayErosion(GrayDilation(image, maskHeight, maskWidth), maskHeight, maskWidth);

    /// <summary>灰階圓形開運算。先用圓侵蝕再膨脹。比圓小的亮點會消失。</summary>
    public static ImageBuffer GrayOpenCircle(ImageBuffer image, int radius) =>
        GrayDilationCircle(GrayErosionCircle(image, radius), radius);

    /// <summary>灰階圓形閉運算。先用圓膨脹再侵蝕。比圓小的暗點會補上。</summary>
    public static ImageBuffer GrayCloseCircle(ImageBuffer image, int radius) =>
        GrayErosionCircle(GrayDilationCircle(image, radius), radius);

    static ImageBuffer ApplyCircle(ImageBuffer image, int radius, bool dark)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (radius < 0)
            throw new ArgumentOutOfRangeException(nameof(radius), "半徑不能是負的");
        byte[] gray = image.GrayBytes();
        var native = new byte[gray.Length];
        if (NativeCore.TryGrayEllipse(gray, image.Width, image.Height, radius, !dark, native))
            return ImageBuffer.Gray(image.Width, image.Height, native);
        byte[] result = FilterOffsets(gray, image.Width, image.Height, EllipseOffsets(radius), dark);
        return ImageBuffer.Gray(image.Width, image.Height, result);
    }

    static ImageBuffer ApplyRectangle(ImageBuffer image, int maskHeight, int maskWidth, bool dark)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        byte[] gray = image.GrayBytes();
        var native = new byte[gray.Length];
        if (NativeCore.TryGrayRect(gray, image.Width, image.Height, maskHeight, maskWidth, !dark, native))
            return ImageBuffer.Gray(image.Width, image.Height, native);
        byte[] result = Filter(gray, image.Width, image.Height, maskHeight, maskWidth, dark);
        return ImageBuffer.Gray(image.Width, image.Height, result);
    }

    static List<(int Row, int Column)> EllipseOffsets(int radius)
    {
        int size = radius * 2 + 1;
        int center = radius;
        double inverse = center == 0 ? 0 : 1.0 / (center * center);
        var offsets = new List<(int Row, int Column)>();
        for (int row = 0; row < size; row++)
        {
            int dy = row - center;
            if (Math.Abs(dy) > center)
                continue;
            int reach = center == 0 ? 0 : (int)Math.Round(center * Math.Sqrt(Math.Max(0, 1.0 - dy * dy * inverse)), MidpointRounding.AwayFromZero);
            int start = Math.Max(center - reach, 0);
            int end = Math.Min(center + reach + 1, size);
            for (int column = start; column < end; column++)
                offsets.Add((row - center, column - center));
        }
        return offsets;
    }

    static byte[] FilterOffsets(byte[] source, int width, int height, List<(int Row, int Column)> offsets, bool dark)
    {
        var result = new byte[source.Length];
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                byte value = dark ? byte.MaxValue : byte.MinValue;
                foreach (var offset in offsets)
                {
                    int sampleRow = row + offset.Row;
                    int sampleColumn = column + offset.Column;
                    byte sample = dark ? byte.MaxValue : byte.MinValue;
                    if ((uint)sampleRow < (uint)height && (uint)sampleColumn < (uint)width)
                        sample = source[sampleRow * width + sampleColumn];
                    if (dark)
                    {
                        if (sample < value)
                            value = sample;
                    }
                    else if (sample > value)
                    {
                        value = sample;
                    }
                }
                result[row * width + column] = value;
            }
        }
        return result;
    }

    static byte[] Filter(byte[] source, int width, int height, int maskHeight, int maskWidth, bool dark)
    {
        int anchorRow = maskHeight / 2;
        int anchorColumn = maskWidth / 2;
        var result = new byte[source.Length];
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                byte value = dark ? byte.MaxValue : byte.MinValue;
                for (int maskRow = 0; maskRow < maskHeight; maskRow++)
                {
                    int sampleRow = row + maskRow - anchorRow;
                    for (int maskColumn = 0; maskColumn < maskWidth; maskColumn++)
                    {
                        int sampleColumn = column + maskColumn - anchorColumn;
                        byte sample = dark ? byte.MaxValue : byte.MinValue;
                        if ((uint)sampleRow < (uint)height && (uint)sampleColumn < (uint)width)
                            sample = source[sampleRow * width + sampleColumn];
                        if (dark)
                        {
                            if (sample < value)
                                value = sample;
                        }
                        else if (sample > value)
                        {
                            value = sample;
                        }
                    }
                }
                result[row * width + column] = value;
            }
        }
        return result;
    }
}
