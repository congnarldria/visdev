namespace Vizion;

public static partial class Operators
{
    public static ImageBuffer LutTrans(ImageBuffer image, double[] table)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (table == null || table.Length != 256)
            throw new ArgumentException("Lut 要 256 個數，對應灰階 0 到 255");
        var mapped = new byte[256];
        for (int index = 0; index < 256; index++)
            mapped[index] = RoundByte(table[index]);
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryLut(source, destination, source.Length, mapped))
                return;
            for (int index = 0; index < source.Length; index++)
                destination[index] = mapped[source[index]];
        });
    }

    public static ImageBuffer Rgb3ToGray(ImageBuffer red, ImageBuffer green, ImageBuffer blue)
    {
        if (red == null || green == null || blue == null)
            throw new ArgumentNullException(nameof(red));
        if (red.Channels != 1 || green.Channels != 1 || blue.Channels != 1)
            throw new ArgumentException("三張都要是灰階");
        if (green.Width != red.Width || blue.Width != red.Width || green.Height != red.Height || blue.Height != red.Height)
            throw new ArgumentException("三張影像的尺寸不同");
        int count = red.Width * red.Height;
        var destination = new byte[count];
        if (!NativeCore.TryRgb3Gray(red.Pixels, green.Pixels, blue.Pixels, destination, count))
        {
            for (int index = 0; index < count; index++)
                destination[index] = RoundD(0.299 * red.Pixels[index] + 0.587 * green.Pixels[index] + 0.114 * blue.Pixels[index]);
        }
        return ImageBuffer.Gray(red.Width, red.Height, destination);
    }

    public static ImageBuffer GenImageProto(ImageBuffer image, double gray)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        int value = RoundByte(gray);
        var destination = new byte[image.Pixels.Length];
        if (!NativeCore.TryFill(destination, destination.Length, value))
        {
            for (int index = 0; index < destination.Length; index++)
                destination[index] = (byte)value;
        }
        return ImageBuffer.Wrap(image.Width, image.Height, image.Channels, destination, null);
    }

    public static ImageBuffer PaintGray(ImageBuffer source, ImageBuffer destination) =>
        Paint(destination, source, source, allowColorOntoGray: false);

    public static ImageBuffer OverpaintGray(ImageBuffer destination, ImageBuffer source) =>
        Paint(destination, source, destination, allowColorOntoGray: true);

    public static ImageBuffer RankRect(ImageBuffer image, int maskWidth, int maskHeight, int rank)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        int wide = OddSide(maskWidth, "MaskWidth", 7);
        int high = OddSide(maskHeight, "MaskHeight", 7);
        int window = wide * high;
        if (rank < 0 || rank >= window)
            throw new ArgumentException("Rank 要在 0 到 " + (window - 1));
        byte[] gray = image.GrayBytes();
        var destination = new byte[gray.Length];
        if (!NativeCore.TryRank(gray, destination, image.Width, image.Height, wide, high, rank, 0))
            RankScalar(gray, destination, image.Width, image.Height, wide, high, rank, 0);
        return ImageBuffer.Gray(image.Width, image.Height, destination);
    }

    public static ImageBuffer MedianRect(ImageBuffer image, int maskWidth, int maskHeight, string margin)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        int wide = OddSide(maskWidth, "MaskWidth", 7);
        int high = OddSide(maskHeight, "MaskHeight", 7);
        int border = MedianBorder(margin);
        byte[] gray = image.GrayBytes();
        var destination = new byte[gray.Length];
        int rank = (wide * high) / 2;
        if (!NativeCore.TryRank(gray, destination, image.Width, image.Height, wide, high, rank, border))
            RankScalar(gray, destination, image.Width, image.Height, wide, high, rank, border);
        return ImageBuffer.Wrap(image.Width, image.Height, 1, destination, image.Domain);
    }

    public static ImageBuffer PrewittAmp(ImageBuffer image)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        return GrayKept(image, (source, destination) =>
        {
            if (NativeCore.TryPrewitt(source, destination, image.Width, image.Height))
                return;
            PrewittScalar(source, destination, image.Width, image.Height);
        });
    }

    public static ImageBuffer Roberts(ImageBuffer image, string filterType)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        bool sumAbs = RobertsAbs(filterType);
        return GrayKept(image, (source, destination) =>
        {
            if (NativeCore.TryRoberts(source, destination, image.Width, image.Height, sumAbs))
                return;
            RobertsScalar(source, destination, image.Width, image.Height, sumAbs);
        });
    }

    public static ImageBuffer BinomialFilter(ImageBuffer image, int maskWidth, int maskHeight)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (maskWidth < 1 || maskHeight < 1 || maskWidth > 31 || maskHeight > 31 || (maskWidth & 1) == 0 || (maskHeight & 1) == 0)
            throw new ArgumentException("遮罩寬高要是 1 到 31 的奇數");
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryBinomial(source, destination, image.Width, image.Height, maskWidth, maskHeight))
                return;
            BinomialScalar(source, destination, image.Width, image.Height, maskWidth, maskHeight);
        });
    }

    static ImageBuffer Paint(ImageBuffer destination, ImageBuffer source, ImageBuffer maskOwner, bool allowColorOntoGray)
    {
        if (destination == null || source == null)
            throw new ArgumentNullException(nameof(destination));
        if (source.Width != destination.Width || source.Height != destination.Height)
            throw new ArgumentException("兩張影像的尺寸不同");
        int sourceChannels = source.Channels;
        byte[] sourcePixels = source.Pixels;
        if (source.Channels != destination.Channels)
        {
            if (source.Channels == 1 && destination.Channels == 3)
                sourceChannels = 1;
            else if (allowColorOntoGray && source.Channels == 3 && destination.Channels == 1)
            {
                sourcePixels = source.GrayBytes();
                sourceChannels = 1;
            }
            else if (allowColorOntoGray)
                throw new ArgumentException("兩張影像的通道數要相同，或其中一張是灰階");
            else
                throw new ArgumentException("來源和目的的通道數要相同，或來源是灰階、目的是彩色");
        }
        byte[]? mask = ByteDomain(maskOwner);
        var output = new byte[destination.Pixels.Length];
        if (!NativeCore.TryPaint(destination.Pixels, sourcePixels, output, destination.Width, destination.Height, destination.Channels, sourceChannels, mask))
            PaintScalar(destination.Pixels, sourcePixels, output, destination.Width, destination.Height, destination.Channels, sourceChannels, mask);
        return ImageBuffer.Wrap(destination.Width, destination.Height, destination.Channels, output, destination.Domain);
    }

    static void PaintScalar(byte[] baseImage, byte[] source, byte[] destination, int width, int height, int baseChannels, int sourceChannels, byte[]? mask)
    {
        Buffer.BlockCopy(baseImage, 0, destination, 0, destination.Length);
        int count = width * height;
        for (int index = 0; index < count; index++)
        {
            if (mask != null && mask[index] == 0)
                continue;
            if (baseChannels == sourceChannels)
            {
                for (int channel = 0; channel < baseChannels; channel++)
                    destination[index * baseChannels + channel] = source[index * sourceChannels + channel];
            }
            else
            {
                destination[index * 3] = source[index];
                destination[index * 3 + 1] = source[index];
                destination[index * 3 + 2] = source[index];
            }
        }
    }

    static ImageBuffer GrayKept(ImageBuffer image, Action<byte[], byte[]> filter)
    {
        byte[] source = image.GrayBytes();
        var destination = new byte[source.Length];
        filter(source, destination);
        return ImageBuffer.Wrap(image.Width, image.Height, 1, destination, image.Domain);
    }

    static int MedianBorder(string margin)
    {
        string name = (margin ?? "").Trim().ToLowerInvariant();
        if (name == "continued")
            return 0;
        if (name == "mirror")
            return 1;
        if (name == "cyclic")
            return 2;
        throw new ArgumentException("Margin 只能是 continued、mirror 或 cyclic");
    }

    static bool RobertsAbs(string filterType)
    {
        string name = (filterType ?? "").Trim().ToLowerInvariant();
        if (name == "sum_sqrt")
            return false;
        if (name == "sum_abs")
            return true;
        throw new ArgumentException("FilterType 只能是 sum_sqrt 或 sum_abs");
    }

    static byte RoundByte(double value)
    {
        double rounded = Math.Round(value, MidpointRounding.ToEven);
        if (rounded < 0)
            return 0;
        if (rounded > 255)
            return 255;
        return (byte)rounded;
    }

    static int SampleIndex(int position, int length, int border)
    {
        if (border == 2)
        {
            int wrapped = position % length;
            if (wrapped < 0)
                wrapped += length;
            return wrapped;
        }
        if (border == 1)
            return FilterReflect(position, length);
        if (position < 0)
            return 0;
        if (position >= length)
            return length - 1;
        return position;
    }

    static void RankScalar(byte[] source, byte[] destination, int width, int height, int maskWidth, int maskHeight, int rank, int border)
    {
        int anchorX = maskWidth / 2;
        int anchorY = maskHeight / 2;
        var samples = new byte[maskWidth * maskHeight];
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                int filled = 0;
                for (int dy = 0; dy < maskHeight; dy++)
                {
                    int yy = SampleIndex(row + dy - anchorY, height, border);
                    for (int dx = 0; dx < maskWidth; dx++)
                    {
                        int xx = SampleIndex(column + dx - anchorX, width, border);
                        byte value = source[yy * width + xx];
                        int index = filled;
                        while (index > 0 && samples[index - 1] > value)
                        {
                            samples[index] = samples[index - 1];
                            index--;
                        }
                        samples[index] = value;
                        filled++;
                    }
                }
                destination[row * width + column] = samples[rank];
            }
        }
    }

    static int Replicate(int position, int length)
    {
        if (position < 0)
            return 0;
        if (position >= length)
            return length - 1;
        return position;
    }

    static void PrewittScalar(byte[] source, byte[] destination, int width, int height)
    {
        float[] horizontal = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
        float[] vertical = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                float gx = Kernel3(source, width, height, row, column, horizontal);
                float gy = Kernel3(source, width, height, row, column, vertical);
                destination[row * width + column] = RoundF((float)Math.Sqrt(gx * gx + gy * gy) / 3f);
            }
        }
    }

    static float Kernel3(byte[] source, int width, int height, int row, int column, float[] kernel)
    {
        float sum = 0f;
        for (int dy = 0; dy < 3; dy++)
        {
            int yy = Replicate(row + dy - 1, height);
            for (int dx = 0; dx < 3; dx++)
            {
                int xx = Replicate(column + dx - 1, width);
                sum += source[yy * width + xx] * kernel[dy * 3 + dx];
            }
        }
        return sum;
    }

    static void RobertsScalar(byte[] source, byte[] destination, int width, int height, bool sumAbs)
    {
        Array.Clear(destination, 0, destination.Length);
        for (int row = 0; row < height - 1; row++)
        {
            for (int column = 0; column < width - 1; column++)
            {
                float first = source[row * width + column] - source[(row + 1) * width + column + 1];
                float second = source[row * width + column + 1] - source[(row + 1) * width + column];
                float strength = sumAbs ? Math.Abs(first) + Math.Abs(second) : (float)Math.Sqrt(first * first + second * second);
                destination[row * width + column] = RoundF(strength);
            }
        }
    }

    static void BinomialScalar(byte[] source, byte[] destination, int width, int height, int maskWidth, int maskHeight)
    {
        float[] across = BinomialKernel(maskWidth);
        float[] along = BinomialKernel(maskHeight);
        var plane = new float[source.Length];
        for (int index = 0; index < source.Length; index++)
            plane[index] = source[index];
        float[] blurred = SepReplicate(plane, width, height, across, along);
        for (int index = 0; index < destination.Length; index++)
            destination[index] = RoundF(blurred[index]);
    }

    static float[] BinomialKernel(int size)
    {
        int degree = size - 1;
        double coeff = 1;
        double sum = 0;
        var raw = new double[size];
        for (int index = 0; index < size; index++)
        {
            raw[index] = coeff;
            sum += coeff;
            coeff = coeff * (degree - index) / (index + 1.0);
        }
        var kernel = new float[size];
        for (int index = 0; index < size; index++)
            kernel[index] = (float)(raw[index] / sum);
        return kernel;
    }

    static float[] SepReplicate(float[] source, int width, int height, float[] across, float[] along)
    {
        int anchorX = across.Length / 2;
        int anchorY = along.Length / 2;
        var horizontal = new float[source.Length];
        var destination = new float[source.Length];
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                float sum = 0f;
                for (int tap = 0; tap < across.Length; tap++)
                    sum += source[row * width + Replicate(column + tap - anchorX, width)] * across[tap];
                horizontal[row * width + column] = sum;
            }
        }
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                float sum = 0f;
                for (int tap = 0; tap < along.Length; tap++)
                    sum += horizontal[Replicate(row + tap - anchorY, height) * width + column] * along[tap];
                destination[row * width + column] = sum;
            }
        }
        return destination;
    }
}
