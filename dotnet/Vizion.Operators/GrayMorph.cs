namespace Vizion;

public static partial class Operators
{
    /// <summary>灰階矩形開運算。先取遮罩裡最暗的，再取最亮的。比遮罩小的亮點會消失。</summary>
    public static ImageBuffer GrayOpenRectangle(ImageBuffer image, int maskHeight, int maskWidth) =>
        GrayRectangle(image, maskHeight, maskWidth, erodeFirst: true);

    /// <summary>灰階矩形閉運算。先取遮罩裡最亮的，再取最暗的。比遮罩小的暗點會補上。</summary>
    public static ImageBuffer GrayCloseRectangle(ImageBuffer image, int maskHeight, int maskWidth) =>
        GrayRectangle(image, maskHeight, maskWidth, erodeFirst: false);

    static ImageBuffer GrayRectangle(ImageBuffer image, int maskHeight, int maskWidth, bool erodeFirst)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (maskHeight < 1 || maskWidth < 1)
            throw new ArgumentOutOfRangeException(nameof(maskHeight), "遮罩至少要 1x1");
        byte[] source = image.GrayBytes();
        byte[] middle = Filter(source, image.Width, image.Height, maskHeight, maskWidth, dark: erodeFirst);
        byte[] result = Filter(middle, image.Width, image.Height, maskHeight, maskWidth, dark: !erodeFirst);
        return ImageBuffer.Gray(image.Width, image.Height, result);
    }

    static byte[] Filter(byte[] source, int width, int height, int maskHeight, int maskWidth, bool dark)
    {
        int anchorRow = (maskHeight - 1) / 2;
        int anchorColumn = (maskWidth - 1) / 2;
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
