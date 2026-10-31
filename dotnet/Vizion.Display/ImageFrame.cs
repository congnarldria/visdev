namespace Vizion;

/// <summary>一張要顯示的影像。建構時會複製像素，呼叫端可以接著覆寫自己的緩衝。</summary>
public sealed class ImageFrame
{
    public int Width { get; }
    public int Height { get; }
    public PixelLayout Layout { get; }
    public int Channels { get; }
    public byte[] Pixels { get; }
    public int Stride { get; }

    private ImageFrame(int width, int height, PixelLayout layout, byte[] pixels, int stride)
    {
        Width = width;
        Height = height;
        Layout = layout;
        Channels = layout == PixelLayout.Gray ? 1 : 3;
        Pixels = pixels;
        Stride = stride;
    }

    public static ImageFrame Gray(int width, int height, byte[] pixels, int stride = 0) =>
        Create(width, height, pixels, PixelLayout.Gray, stride);

    public static ImageFrame Create(int width, int height, byte[] pixels, PixelLayout layout, int stride = 0)
    {
        if (pixels == null)
            throw new ArgumentNullException(nameof(pixels));
        if (width <= 0 || height <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), "寬高至少要 1");
        int channels = layout == PixelLayout.Gray ? 1 : 3;
        int tight = width * channels;
        if (stride == 0)
            stride = tight;
        if (stride < tight)
            throw new ArgumentException("stride 比一列的位元組數還小");
        long need = (long)stride * (height - 1) + tight;
        if (pixels.Length < need)
            throw new ArgumentException("像素緩衝比寬高還短");
        var copy = new byte[tight * height];
        for (int row = 0; row < height; row++)
            Buffer.BlockCopy(pixels, row * stride, copy, row * tight, tight);
        return new ImageFrame(width, height, layout, copy, tight);
    }

    public int GrayAt(int row, int column)
    {
        if ((uint)row >= (uint)Height || (uint)column >= (uint)Width)
            return -1;
        int index = row * Stride + column * Channels;
        if (Channels == 1)
            return Pixels[index];
        int b, g, r;
        if (Layout == PixelLayout.Bgr)
        {
            b = Pixels[index];
            g = Pixels[index + 1];
            r = Pixels[index + 2];
        }
        else
        {
            r = Pixels[index];
            g = Pixels[index + 1];
            b = Pixels[index + 2];
        }
        return (r * 299 + g * 587 + b * 114) / 1000;
    }
}
