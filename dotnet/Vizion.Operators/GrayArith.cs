namespace Vizion;

public static partial class Operators
{
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

    public static ImageBuffer ScaleImage(ImageBuffer image, double mult, double add)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        return Map(image, (source, destination) => Apply(source, null, destination, 1, mult, add));
    }

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

    public static ImageBuffer AddImage(ImageBuffer left, ImageBuffer right, double mult, double add) =>
        Combine(left, right, 2, mult, add);

    public static ImageBuffer SubImage(ImageBuffer left, ImageBuffer right, double mult, double add) =>
        Combine(left, right, 3, mult, add);

    public static ImageBuffer MultImage(ImageBuffer left, ImageBuffer right, double mult, double add) =>
        Combine(left, right, 4, mult, add);

    public static ImageBuffer AbsDiffImage(ImageBuffer left, ImageBuffer right, double mult) =>
        Combine(left, right, 5, mult, 0);

    public static ImageBuffer MinImage(ImageBuffer left, ImageBuffer right) =>
        Combine(left, right, 6, 1, 0);

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
