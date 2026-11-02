namespace Vizion;

/// <summary>讀進來的影像。灰階一通道，彩色是 BGR，和 OpenCV 相同。像素在建構時複製。</summary>
public sealed class ImageBuffer
{
    public int Width { get; }
    public int Height { get; }
    public int Channels { get; }
    public byte[] Pixels { get; }
    public PixelLayout Layout => Channels == 1 ? PixelLayout.Gray : PixelLayout.Bgr;

    private ImageBuffer(int width, int height, int channels, byte[] pixels)
    {
        Width = width;
        Height = height;
        Channels = channels;
        Pixels = pixels;
    }

    public static ImageBuffer Gray(int width, int height, byte[] pixels) => From(width, height, 1, pixels);

    public static ImageBuffer Bgr(int width, int height, byte[] pixels) => From(width, height, 3, pixels);

    public static ImageBuffer From(int width, int height, int channels, byte[] pixels)
    {
        if (pixels == null)
            throw new ArgumentNullException(nameof(pixels));
        if (width <= 0 || height <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), "寬高至少要 1");
        if (channels != 1 && channels != 3)
            throw new ArgumentOutOfRangeException(nameof(channels), "通道只能是 1 或 3");
        int need = checked(width * height * channels);
        if (pixels.Length < need)
            throw new ArgumentException("像素緩衝比寬高還短");
        var copy = new byte[need];
        Buffer.BlockCopy(pixels, 0, copy, 0, need);
        return new ImageBuffer(width, height, channels, copy);
    }

    internal static ImageBuffer Wrap(int width, int height, int channels, byte[] pixels) =>
        new(width, height, channels, pixels);

    /// <summary>灰階影像回傳自己的緩衝。彩色先轉成和 threshold 相同的灰階。</summary>
    public byte[] GrayBytes()
    {
        if (Channels == 1)
            return Pixels;
        var gray = new byte[Width * Height];
        for (int i = 0, p = 0; i < gray.Length; i++, p += 3)
            gray[i] = (byte)GrayConvert.Bgr(Pixels[p], Pixels[p + 1], Pixels[p + 2]);
        return gray;
    }
}
