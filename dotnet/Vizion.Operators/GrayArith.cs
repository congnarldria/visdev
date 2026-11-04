namespace Vizion;

public static partial class Operators
{
    /// <summary>彩色影像轉成單通道。已經是灰階就直接複製。</summary>
    /// <remarks>轉成灰階</remarks>
    /// <example>
    /// <code>Vizion.rgb1_to_gray (Image, Gray)</code>
    /// </example>
    /// <param name="image">彩色或灰階影像。</param>
    /// <returns>單通道灰階。已經是灰階就直接複製。</returns>
    public static ImageBuffer Rgb1ToGray(ImageBuffer image)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (image.Channels == 1)
        {
            var copy = new byte[image.Pixels.Length];
            Buffer.BlockCopy(image.Pixels, 0, copy, 0, copy.Length);
            return ImageBuffer.Wrap(image.Width, image.Height, 1, copy, image.Domain);
        }
        return ImageBuffer.Wrap(image.Width, image.Height, 1, image.GrayBytes(), image.Domain);
    }

    /// <summary>每個像素變成 255 減去原值。</summary>
    /// <remarks>灰階反相</remarks>
    /// <example>
    /// <code>Vizion.invert_image (Image, Inverted)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <returns>每個像素變成 255 減去原值。</returns>
    public static ImageBuffer InvertImage(ImageBuffer image)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryGrayArith(source, null, destination, source.Length, 0, 0, 0))
                return;
            for (int i = 0; i < source.Length; i++)
                destination[i] = (byte)(255 - source[i]);
        });
    }

    /// <summary>輸出 = 輸入 × Mult + Add，超過 0 到 255 的值會被截斷。</summary>
    /// <remarks>線性縮放灰階</remarks>
    /// <example>
    /// <code>Vizion.scale_image (Image, Scaled, 1.2, 0)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="mult">乘上的係數。</param>
    /// <param name="add">加上的偏移。</param>
    /// <returns>縮放後的影像。超過 0 到 255 會截斷。</returns>
    public static ImageBuffer ScaleImage(ImageBuffer image, double mult, double add)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        return Map(image, (source, destination) => Apply(source, null, destination, 1, mult, add));
    }

    /// <summary>這版只收半徑。HALCON 的 MaskType 與 Margin 先省略。</summary>
    /// <remarks>中值濾波</remarks>
    /// <example>
    /// <code>Vizion.median_image (Image, Median, 2)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="radius">半徑，像素。</param>
    /// <returns>中值濾波結果。</returns>
    public static ImageBuffer MedianImage(ImageBuffer image, int radius)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        int kernel = Math.Max(1, radius) * 2 + 1;
        if ((kernel & 1) == 0)
            kernel++;
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryMedian(source, destination, image.Width, image.Height, kernel))
                return;
            MedianScalar(source, destination, image.Width, image.Height, kernel);
        });
    }

    /// <summary>每個像素是 (灰階1 + 灰階2) × Mult + Add，超過 0 到 255 會截斷。兩張圖的尺寸和通道數要相同。domain 取交集，外面補 0。</summary>
    /// <remarks>兩張影像相加</remarks>
    /// <example>
    /// <code>Vizion.add_image (Image1, Image2, ImageResult, 0.5, 0)</code>
    /// </example>
    /// <param name="left">第一張影像。</param>
    /// <param name="right">第二張影像。</param>
    /// <param name="mult">乘上的係數。</param>
    /// <param name="add">再加上的值。</param>
    /// <returns>相加結果。定義域是兩張圖的交集，外面是 0。</returns>
    public static ImageBuffer AddImage(ImageBuffer left, ImageBuffer right, double mult, double add) =>
        Combine(left, right, 2, mult, add);

    /// <summary>每個像素是 (被減影像 - 減數影像) × Mult + Add，超過 0 到 255 會截斷。</summary>
    /// <remarks>兩張影像相減</remarks>
    /// <example>
    /// <code>Vizion.sub_image (Image1, Image2, ImageSub, 1, 128)</code>
    /// </example>
    /// <param name="left">被減影像。</param>
    /// <param name="right">減去的影像。</param>
    /// <param name="mult">乘上的係數。</param>
    /// <param name="add">再加上的值。</param>
    /// <returns>相減結果。定義域是兩張圖的交集，外面是 0。</returns>
    public static ImageBuffer SubImage(ImageBuffer left, ImageBuffer right, double mult, double add) =>
        Combine(left, right, 3, mult, add);

    /// <summary>每個像素是 (灰階1 × 灰階2) × Mult + Add。灰階直接相乘會很大，Mult 通常要小於 1。</summary>
    /// <remarks>兩張影像相乘</remarks>
    /// <example>
    /// <code>Vizion.mult_image (Image1, Image2, ImageResult, 0.005, 0)</code>
    /// </example>
    /// <param name="left">第一張影像。</param>
    /// <param name="right">第二張影像。</param>
    /// <param name="mult">乘上的係數。</param>
    /// <param name="add">再加上的值。</param>
    /// <returns>相乘結果。定義域是兩張圖的交集，外面是 0。</returns>
    public static ImageBuffer MultImage(ImageBuffer left, ImageBuffer right, double mult, double add) =>
        Combine(left, right, 4, mult, add);

    /// <summary>每個像素是 |灰階1 - 灰階2| × Mult，超過 255 會截斷。</summary>
    /// <remarks>兩張影像的絕對差</remarks>
    /// <example>
    /// <code>Vizion.abs_diff_image (Image1, Image2, ImageAbsDiff, 1)</code>
    /// </example>
    /// <param name="left">第一張影像。</param>
    /// <param name="right">第二張影像。</param>
    /// <param name="mult">乘上的係數。</param>
    /// <returns>絕對差。定義域是兩張圖的交集，外面是 0。</returns>
    public static ImageBuffer AbsDiffImage(ImageBuffer left, ImageBuffer right, double mult) =>
        Combine(left, right, 5, mult, 0);

    /// <summary>每個像素是 (灰階1 / 灰階2) × Mult + Add。分母是 0 時商先當 0，再乘上 Mult 加上 Add，所以結果是 Add。兩張圖的尺寸和通道數要相同。定義域取交集，外面補 0。</summary>
    /// <remarks>兩張影像相除</remarks>
    /// <example>
    /// <code>Vizion.div_image (Image1, Image2, ImageResult, 1, 0)</code>
    /// </example>
    /// <param name="left">分子影像。</param>
    /// <param name="right">分母影像。</param>
    /// <param name="mult">乘上的係數。</param>
    /// <param name="add">再加上的值。</param>
    /// <returns>相除結果。定義域是兩張圖的交集，外面是 0。</returns>
    public static ImageBuffer DivImage(ImageBuffer left, ImageBuffer right, double mult, double add) =>
        Combine(left, right, 8, mult, add);

    /// <summary>兩張圖的尺寸和通道數要相同。</summary>
    /// <remarks>每個像素取較小灰階</remarks>
    /// <example>
    /// <code>Vizion.min_image (Image1, Image2, ImageMin)</code>
    /// </example>
    /// <param name="left">第一張影像。</param>
    /// <param name="right">第二張影像。</param>
    /// <returns>每個像素的較小值。定義域是兩張圖的交集，外面是 0。</returns>
    public static ImageBuffer MinImage(ImageBuffer left, ImageBuffer right) =>
        Combine(left, right, 6, 1, 0);

    /// <summary>兩張圖的尺寸和通道數要相同。</summary>
    /// <remarks>每個像素取較大灰階</remarks>
    /// <example>
    /// <code>Vizion.max_image (Image1, Image2, ImageMax)</code>
    /// </example>
    /// <param name="left">第一張影像。</param>
    /// <param name="right">第二張影像。</param>
    /// <returns>每個像素的較大值。定義域是兩張圖的交集，外面是 0。</returns>
    public static ImageBuffer MaxImage(ImageBuffer left, ImageBuffer right) =>
        Combine(left, right, 7, 1, 0);

    static ImageBuffer Combine(ImageBuffer left, ImageBuffer right, int op, double mult, double add)
    {
        if (left == null || right == null)
            throw new ArgumentNullException(nameof(left));
        if (left.Width != right.Width || left.Height != right.Height || left.Channels != right.Channels)
            throw new ArgumentException("兩張影像的尺寸或通道數不同");
        var destination = new byte[left.Pixels.Length];
        Apply(left.Pixels, right.Pixels, destination, op, mult, add);
        Region? domain = AndDomain(left, right);
        if (domain != null)
            ZeroOutside(destination, left.Width, left.Height, left.Channels, domain);
        return ImageBuffer.Wrap(left.Width, left.Height, left.Channels, destination, domain);
    }

    static void Apply(byte[] left, byte[]? right, byte[] destination, int op, double mult, double add)
    {
        if (NativeCore.TryGrayArith(left, right, destination, left.Length, op, mult, add))
            return;
        for (int i = 0; i < left.Length; i++)
        {
            float a = left[i];
            float b = right == null ? 0 : right[i];
            float value;
            switch (op)
            {
                case 0:
                    destination[i] = (byte)(255 - left[i]);
                    continue;
                case 1:
                    value = a * (float)mult + (float)add;
                    break;
                case 2:
                    value = (a + b) * (float)mult + (float)add;
                    break;
                case 3:
                    value = (a - b) * (float)mult + (float)add;
                    break;
                case 4:
                    value = (a * b) * (float)mult + (float)add;
                    break;
                case 5:
                    value = Math.Abs(a - b) * (float)mult + (float)add;
                    break;
                case 8:
                    value = (b == 0f ? 0f : a / b) * (float)mult + (float)add;
                    break;
                case 9:
                    value = a + (a - b) * (float)mult;
                    break;
                case 10:
                    value = (a - b) + b * (float)mult + (float)add;
                    break;
                case 6:
                    destination[i] = left[i] < right![i] ? left[i] : right[i];
                    continue;
                default:
                    destination[i] = left[i] > right![i] ? left[i] : right[i];
                    continue;
            }
            destination[i] = Trunc(value);
        }
    }

    static byte Trunc(float value)
    {
        if (value <= 0f)
            return 0;
        if (value >= 255f)
            return 255;
        return (byte)value;
    }

    static ImageBuffer Map(ImageBuffer image, Action<byte[], byte[]> filter)
    {
        if (image.Channels == 1)
        {
            var destination = new byte[image.Pixels.Length];
            filter(image.Pixels, destination);
            return ImageBuffer.Wrap(image.Width, image.Height, 1, destination, image.Domain);
        }
        int plane = image.Width * image.Height;
        var merged = new byte[plane * 3];
        var source = new byte[plane];
        var planeOut = new byte[plane];
        for (int channel = 0; channel < 3; channel++)
        {
            for (int index = 0; index < plane; index++)
                source[index] = image.Pixels[index * 3 + channel];
            filter(source, planeOut);
            for (int index = 0; index < plane; index++)
                merged[index * 3 + channel] = planeOut[index];
        }
        return ImageBuffer.Wrap(image.Width, image.Height, 3, merged, image.Domain);
    }

    static void MedianScalar(byte[] source, byte[] destination, int width, int height, int ksize)
    {
        if (ksize <= 1)
        {
            Buffer.BlockCopy(source, 0, destination, 0, source.Length);
            return;
        }
        int radius = ksize / 2;
        int window = ksize * ksize;
        var values = new byte[window];
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                int filled = 0;
                for (int dy = -radius; dy <= radius; dy++)
                {
                    int yy = row + dy;
                    if (yy < 0)
                        yy = 0;
                    else if (yy >= height)
                        yy = height - 1;
                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        int xx = column + dx;
                        if (xx < 0)
                            xx = 0;
                        else if (xx >= width)
                            xx = width - 1;
                        byte value = source[yy * width + xx];
                        int index = filled;
                        while (index > 0 && values[index - 1] > value)
                        {
                            values[index] = values[index - 1];
                            index--;
                        }
                        values[index] = value;
                        filled++;
                    }
                }
                destination[row * width + column] = values[window / 2];
            }
        }
    }

    static Region? AndDomain(ImageBuffer left, ImageBuffer right)
    {
        if (left.Domain == null && right.Domain == null)
            return null;
        Region first = left.Domain ?? Full(left.Width, left.Height);
        Region second = right.Domain ?? Full(right.Width, right.Height);
        return Intersection(first, second);
    }

    static Region Full(int width, int height)
    {
        var rows = new int[height];
        var columns = new int[height];
        var lengths = new int[height];
        for (int row = 0; row < height; row++)
        {
            rows[row] = row;
            lengths[row] = width;
        }
        return new Region(width, height, rows, columns, lengths);
    }

    static void ZeroOutside(byte[] pixels, int width, int height, int channels, Region domain)
    {
        var inside = new bool[width * height];
        for (int i = 0; i < domain.Rows.Length; i++)
        {
            int row = domain.Rows[i];
            int column = domain.Columns[i];
            int length = domain.Lengths[i];
            if ((uint)row >= (uint)height || length <= 0)
                continue;
            if (column < 0)
            {
                length += column;
                column = 0;
            }
            if (column >= width)
                continue;
            if (column + length > width)
                length = width - column;
            int offset = row * width + column;
            for (int k = 0; k < length; k++)
                inside[offset + k] = true;
        }
        for (int index = 0; index < inside.Length; index++)
        {
            if (inside[index])
                continue;
            int pixel = index * channels;
            for (int channel = 0; channel < channels; channel++)
                pixels[pixel + channel] = 0;
        }
    }
}
