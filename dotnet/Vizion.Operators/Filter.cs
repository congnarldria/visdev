namespace Vizion;

public static partial class Operators
{
    /// <summary>矩形平均。邊界用鏡射，和 cv2.blur 相同。遮罩至少 1x1。</summary>
    public static ImageBuffer MeanImage(ImageBuffer image, int maskWidth, int maskHeight)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (maskWidth < 1 || maskHeight < 1)
            throw new ArgumentOutOfRangeException(nameof(maskWidth), "遮罩至少要 1x1");
        return FilterPlanes(image, (source, destination) =>
        {
            if (NativeCore.TryMean(source, destination, image.Width, image.Height, maskWidth, maskHeight))
                return;
            MeanScalar(source, destination, image.Width, image.Height, maskWidth, maskHeight);
        });
    }

    /// <summary>高斯平滑。Size 是大於等於 3 的奇數，和 cv2.GaussianBlur 的 sigma 0 相同。</summary>
    public static ImageBuffer GaussImage(ImageBuffer image, int size)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (size < 3 || (size & 1) == 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size 必須是大於等於 3 的奇數");
        return FilterPlanes(image, (source, destination) =>
        {
            if (NativeCore.TryGauss(source, destination, image.Width, image.Height, size))
                return;
            GaussScalar(source, destination, image.Width, image.Height, size);
        });
    }

    static ImageBuffer FilterPlanes(ImageBuffer image, Action<byte[], byte[]> filter)
    {
        if (image.Channels == 1)
        {
            var destination = new byte[image.Pixels.Length];
            filter(image.Pixels, destination);
            return ImageBuffer.Gray(image.Width, image.Height, destination);
        }
        int plane = image.Width * image.Height;
        var merged = new byte[plane * 3];
        var planeSource = new byte[plane];
        var planeDestination = new byte[plane];
        for (int channel = 0; channel < 3; channel++)
        {
            for (int index = 0; index < plane; index++)
                planeSource[index] = image.Pixels[index * 3 + channel];
            filter(planeSource, planeDestination);
            for (int index = 0; index < plane; index++)
                merged[index * 3 + channel] = planeDestination[index];
        }
        return ImageBuffer.Bgr(image.Width, image.Height, merged);
    }

    static int FilterReflect(int position, int length)
    {
        if (length <= 1)
            return 0;
        while (position < 0 || position >= length)
        {
            if (position < 0)
                position = -position;
            else
                position = length * 2 - position - 2;
        }
        return position;
    }

    static int MeanDiv(uint sum, int area)
    {
        if (area <= 1)
            return (int)sum;
        if ((area & (area - 1)) == 0)
            return (int)((sum + (uint)(area / 2 + 1)) / (uint)area);
        return (int)((double)sum / area + 0.5);
    }

    static void MeanScalar(byte[] source, byte[] destination, int width, int height, int maskWidth, int maskHeight)
    {
        int anchorX = maskWidth / 2;
        int anchorY = maskHeight / 2;
        int area = maskWidth * maskHeight;
        var horizontal = new uint[width * height];
        for (int row = 0; row < height; row++)
        {
            int offset = row * width;
            uint sum = 0;
            for (int tap = 0; tap < maskWidth; tap++)
                sum += source[offset + FilterReflect(tap - anchorX, width)];
            horizontal[offset] = sum;
            for (int column = 1; column < width; column++)
            {
                sum += source[offset + FilterReflect(column + maskWidth - 1 - anchorX, width)];
                sum -= source[offset + FilterReflect(column - 1 - anchorX, width)];
                horizontal[offset + column] = sum;
            }
        }
        for (int column = 0; column < width; column++)
        {
            uint sum = 0;
            for (int tap = 0; tap < maskHeight; tap++)
                sum += horizontal[FilterReflect(tap - anchorY, height) * width + column];
            destination[column] = (byte)MeanDiv(sum, area);
            for (int row = 1; row < height; row++)
            {
                sum += horizontal[FilterReflect(row + maskHeight - 1 - anchorY, height) * width + column];
                sum -= horizontal[FilterReflect(row - 1 - anchorY, height) * width + column];
                destination[row * width + column] = (byte)MeanDiv(sum, area);
            }
        }
    }

    static int[] GaussKernel(int size, out int scale)
    {
        int[] kernel = size switch
        {
            3 => new[] { 64, 128, 64 },
            5 => new[] { 16, 64, 96, 64, 16 },
            7 => new[] { 8, 28, 56, 72, 56, 28, 8 },
            9 => new[] { 4, 13, 30, 51, 60, 51, 30, 13, 4 },
            11 => new[] { 2, 7, 17, 31, 45, 52, 45, 31, 17, 7, 2 },
            13 => new[] { 1, 5, 10, 19, 30, 41, 44, 41, 30, 19, 10, 5, 1 },
            15 => new[] { 1, 3, 6, 12, 20, 30, 36, 40, 36, 30, 20, 12, 6, 3, 1 },
            _ => GaussFormula(size),
        };
        scale = 0;
        for (int i = 0; i < kernel.Length; i++)
            scale += kernel[i];
        return kernel;
    }

    static int[] GaussFormula(int size)
    {
        double sigma = 0.3 * (((size - 1) * 0.5) - 1.0) + 0.8;
        double inverse = -0.5 / (sigma * sigma);
        var raw = new double[size];
        double sum = 0;
        for (int i = 0; i < size; i++)
        {
            double axis = i - (size - 1) * 0.5;
            raw[i] = Math.Exp(inverse * axis * axis);
            sum += raw[i];
        }
        var kernel = new int[size];
        int total = 0;
        for (int i = 0; i < size; i++)
        {
            kernel[i] = (int)(raw[i] / sum * 256.0 + 0.5);
            total += kernel[i];
        }
        kernel[size / 2] += 256 - total;
        return kernel;
    }

    static void GaussScalar(byte[] source, byte[] destination, int width, int height, int size)
    {
        int[] kernel = GaussKernel(size, out int scale);
        int anchor = size / 2;
        long divisor = (long)scale * scale;
        var horizontal = new long[width * height];
        for (int row = 0; row < height; row++)
        {
            int offset = row * width;
            for (int column = 0; column < width; column++)
            {
                long sum = 0;
                for (int tap = 0; tap < size; tap++)
                    sum += (long)source[offset + FilterReflect(column + tap - anchor, width)] * kernel[tap];
                horizontal[offset + column] = sum;
            }
        }
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                long sum = 0;
                for (int tap = 0; tap < size; tap++)
                    sum += horizontal[FilterReflect(row + tap - anchor, height) * width + column] * kernel[tap];
                int value = (int)((sum + divisor / 2) / divisor);
                if (value < 0)
                    value = 0;
                if (value > 255)
                    value = 255;
                destination[row * width + column] = (byte)value;
            }
        }
    }
}
