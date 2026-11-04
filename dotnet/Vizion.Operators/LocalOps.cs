namespace Vizion;

public static partial class Operators
{
    public static ImageBuffer BitNot(ImageBuffer image)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryGrayBitConst(source, destination, source.Length, 0, 0))
                return;
            for (int index = 0; index < source.Length; index++)
                destination[index] = (byte)~source[index];
        });
    }

    public static ImageBuffer BitMask(ImageBuffer image, int mask)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (mask < 0 || mask > 255)
            throw new ArgumentException("BitMask 要在 0 到 255");
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryGrayBitConst(source, destination, source.Length, 1, mask))
                return;
            for (int index = 0; index < source.Length; index++)
                destination[index] = (byte)(source[index] & mask);
        });
    }

    public static ImageBuffer BitSlice(ImageBuffer image, int bit)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (bit < 0 || bit > 7)
            throw new ArgumentException("Bit 要在 0 到 7");
        int flag = 1 << bit;
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryGrayBitConst(source, destination, source.Length, 4, bit))
                return;
            for (int index = 0; index < source.Length; index++)
                destination[index] = (source[index] & flag) != 0 ? (byte)255 : (byte)0;
        });
    }

    public static ImageBuffer BitLShift(ImageBuffer image, int shift) => Shift(image, shift, left: true);

    public static ImageBuffer BitRShift(ImageBuffer image, int shift) => Shift(image, shift, left: false);

    public static ImageBuffer HighpassImage(ImageBuffer image, int width, int height)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        int wide = OddSide(width, "Width", 31);
        int high = OddSide(height, "Height", 31);
        return GrayFilter(image, (source, destination) =>
        {
            if (NativeCore.TryHighpass(source, destination, image.Width, image.Height, wide, high))
                return;
            float[] plane = BytesToFloat(source);
            float[] mean = BoxFloat(plane, image.Width, image.Height, wide, high);
            for (int index = 0; index < source.Length; index++)
                destination[index] = RoundF(128f + source[index] - mean[index]);
        });
    }

    public static ImageBuffer DeviationImage(ImageBuffer image, int width, int height)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (width < 1 || height < 1)
            throw new ArgumentException("遮罩寬高至少要 1");
        return GrayFilter(image, (source, destination) =>
        {
            if (NativeCore.TryDeviation(source, destination, image.Width, image.Height, width, height))
                return;
            float[] plane = BytesToFloat(source);
            float[] mean = BoxFloat(plane, image.Width, image.Height, width, height);
            var squares = new float[plane.Length];
            for (int index = 0; index < plane.Length; index++)
                squares[index] = plane[index] * plane[index];
            float[] square = BoxFloat(squares, image.Width, image.Height, width, height);
            for (int index = 0; index < source.Length; index++)
            {
                float variance = square[index] - mean[index] * mean[index];
                if (variance < 0f)
                    variance = 0f;
                destination[index] = RoundF((float)Math.Sqrt(variance));
            }
        });
    }

    public static ImageBuffer SigmaImage(ImageBuffer image, int width, int height) =>
        DeviationImage(image, width, height);

    public static ImageBuffer GrayRangeRect(ImageBuffer image, int maskHeight, int maskWidth)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (maskHeight < 1 || maskWidth < 1)
            throw new ArgumentException("遮罩寬高至少要 1");
        return GrayFilter(image, (source, destination) =>
        {
            if (NativeCore.TryGraySpan(source, destination, image.Width, image.Height, maskHeight, maskWidth))
                return;
            var gray = ImageBuffer.Gray(image.Width, image.Height, source);
            byte[] bright = GrayDilation(gray, maskHeight, maskWidth).Pixels;
            byte[] dark = GrayErosion(gray, maskHeight, maskWidth).Pixels;
            for (int index = 0; index < destination.Length; index++)
                destination[index] = (byte)(bright[index] - dark[index]);
        });
    }

    public static ImageBuffer Laplace(ImageBuffer image, string resultType, int maskSize, string filterMask)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        bool absolute = LaplaceAbsolute(resultType);
        if (maskSize != 3 && maskSize != 5 && maskSize != 7)
            throw new ArgumentException("MaskSize 只能是 3、5 或 7");
        bool eight = LaplaceEight(filterMask);
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryLaplace(source, destination, image.Width, image.Height, maskSize, eight, absolute))
                return;
            LaplaceScalar(source, destination, image.Width, image.Height, maskSize, eight, absolute);
        });
    }

    static ImageBuffer Shift(ImageBuffer image, int shift, bool left)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (shift < 0)
            throw new ArgumentException("Shift 不能是負的");
        int op = left ? 2 : 3;
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryGrayBitConst(source, destination, source.Length, op, shift))
                return;
            for (int index = 0; index < source.Length; index++)
            {
                if (shift >= 8)
                    destination[index] = 0;
                else if (left)
                    destination[index] = (byte)(source[index] << shift);
                else
                    destination[index] = (byte)(source[index] >> shift);
            }
        });
    }

    static ImageBuffer GrayFilter(ImageBuffer image, Action<byte[], byte[]> filter)
    {
        byte[] source = image.GrayBytes();
        var destination = new byte[source.Length];
        filter(source, destination);
        return ImageBuffer.Wrap(image.Width, image.Height, 1, destination, image.Domain);
    }

    static float[] BytesToFloat(byte[] source)
    {
        var plane = new float[source.Length];
        for (int index = 0; index < source.Length; index++)
            plane[index] = source[index];
        return plane;
    }

    static int OddSide(int value, string name, int limit)
    {
        int side = value | 1;
        if (side < 1 || side > limit)
            throw new ArgumentException(name + " 要在 1 到 " + limit);
        return side;
    }

    static bool LaplaceAbsolute(string resultType)
    {
        string name = (resultType ?? "").Trim().ToLowerInvariant();
        if (name == "absolute")
            return true;
        if (name == "signed")
            return false;
        throw new ArgumentException("ResultType 只能是 absolute 或 signed");
    }

    static bool LaplaceEight(string filterMask)
    {
        string name = (filterMask ?? "").Trim().ToLowerInvariant();
        if (name == "n_4")
            return false;
        if (name == "n_8")
            return true;
        throw new ArgumentException("FilterMask 只能是 n_4 或 n_8");
    }

    static void LaplaceScalar(byte[] source, byte[] destination, int width, int height, int ksize, bool eight, bool absolute)
    {
        var plane = BytesToFloat(source);
        float[] response = ksize == 3
            ? Conv3(plane, width, height, eight)
            : LaplaceWide(plane, width, height, ksize);
        for (int index = 0; index < destination.Length; index++)
        {
            float value = absolute ? Math.Abs(response[index]) : response[index] + 128f;
            destination[index] = Trunc(value);
        }
    }

    static float[] Conv3(float[] source, int width, int height, bool eight)
    {
        float[] kernel = eight
            ? new float[] { 1, 1, 1, 1, -8, 1, 1, 1, 1 }
            : new float[] { 0, 1, 0, 1, -4, 1, 0, 1, 0 };
        var destination = new float[source.Length];
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                float sum = 0f;
                for (int dy = 0; dy < 3; dy++)
                {
                    int yy = FilterReflect(row + dy - 1, height);
                    for (int dx = 0; dx < 3; dx++)
                    {
                        int xx = FilterReflect(column + dx - 1, width);
                        sum += source[yy * width + xx] * kernel[dy * 3 + dx];
                    }
                }
                destination[row * width + column] = sum;
            }
        }
        return destination;
    }

    static float[] LaplaceWide(float[] source, int width, int height, int ksize)
    {
        float[] second = ksize == 5
            ? new float[] { 1, 0, -2, 0, 1 }
            : new float[] { 1, 2, -1, -4, -1, 2, 1 };
        float[] smooth = ksize == 5
            ? new float[] { 1, 4, 6, 4, 1 }
            : new float[] { 1, 6, 15, 20, 15, 6, 1 };
        float[] across = SepFilter(source, width, height, second, smooth);
        float[] along = SepFilter(source, width, height, smooth, second);
        for (int index = 0; index < across.Length; index++)
            across[index] += along[index];
        return across;
    }

    static float[] SepFilter(float[] source, int width, int height, float[] across, float[] along)
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
                    sum += source[row * width + FilterReflect(column + tap - anchorX, width)] * across[tap];
                horizontal[row * width + column] = sum;
            }
        }
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                float sum = 0f;
                for (int tap = 0; tap < along.Length; tap++)
                    sum += horizontal[FilterReflect(row + tap - anchorY, height) * width + column] * along[tap];
                destination[row * width + column] = sum;
            }
        }
        return destination;
    }
}
