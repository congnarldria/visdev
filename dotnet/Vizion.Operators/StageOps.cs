namespace Vizion;

public static partial class Operators
{
    public static ImageBuffer GrayRamp(double alpha, double beta, double gamma, double row, double column, int width, int height)
    {
        if (width < 1 || height < 1)
            throw new ArgumentException("寬高至少要 1");
        var destination = new byte[width * height];
        if (!NativeCore.TryGrayRamp(destination, width, height, alpha, beta, gamma, row, column))
        {
            float a = (float)alpha;
            float b = (float)beta;
            float c = (float)gamma;
            float originRow = (float)row;
            float originColumn = (float)column;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                    destination[y * width + x] = RoundF(a * (y - originRow) + b * (x - originColumn) + c);
            }
        }
        return ImageBuffer.Gray(width, height, destination);
    }

    public static ImageBuffer TileImages(ImageBuffer left, ImageBuffer right, string mode)
    {
        if (left == null || right == null)
            throw new ArgumentNullException(nameof(left));
        if (left.Channels != right.Channels)
            throw new ArgumentException("兩張影像的通道數要相同");
        bool vertical = TileVertical(mode);
        int width = vertical ? Math.Max(left.Width, right.Width) : left.Width + right.Width;
        int height = vertical ? left.Height + right.Height : Math.Max(left.Height, right.Height);
        var destination = new byte[width * height * left.Channels];
        if (!NativeCore.TryTile(left.Pixels, left.Width, left.Height, right.Pixels, right.Width, right.Height, destination, left.Channels, vertical))
            TileScalar(left, right, destination, width, vertical);
        return ImageBuffer.Wrap(width, height, left.Channels, destination, null);
    }

    public static ImageBuffer ConvolImage(ImageBuffer image, string? preset, double[]? weights, int maskWidth, string margin)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        float[] kernel = ConvolutionKernel(preset, weights, maskWidth, out int kernelWidth, out int kernelHeight);
        int border = ConvolutionBorder(margin);
        byte[] gray = image.GrayBytes();
        var destination = new byte[gray.Length];
        if (!NativeCore.TryConvol(gray, destination, image.Width, image.Height, kernel, kernelHeight, kernelWidth, border))
            ConvolScalar(gray, destination, image.Width, image.Height, kernel, kernelHeight, kernelWidth, border);
        return ImageBuffer.Gray(image.Width, image.Height, destination);
    }

    public static ImageBuffer AsinImage(ImageBuffer image) => InverseWave(image, 3);

    public static ImageBuffer AcosImage(ImageBuffer image) => InverseWave(image, 4);

    public static ImageBuffer AtanImage(ImageBuffer image) => InverseWave(image, 5);

    public static ImageBuffer BandpassImage(ImageBuffer image, int inner, int outer)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        int small = OddSide(inner, "Inner", 31);
        int large = OddSide(outer, "Outer", 31);
        if (small >= large)
            throw new ArgumentException("Inner 要比 Outer 小");
        return CompassGray(image, (source, destination) =>
        {
            if (NativeCore.TryBandpass(source, destination, image.Width, image.Height, small, large))
                return;
            float[] plane = new float[source.Length];
            for (int index = 0; index < source.Length; index++)
                plane[index] = source[index];
            float[] narrow = BoxFloat(plane, image.Width, image.Height, small, small);
            float[] wide = BoxFloat(plane, image.Width, image.Height, large, large);
            for (int index = 0; index < destination.Length; index++)
                destination[index] = RoundF(128f + narrow[index] - wide[index]);
        });
    }

    public static ImageBuffer AddImageBorder(ImageBuffer image, int border, int gray)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (border < 0 || gray < 0 || gray > 255)
            throw new ArgumentException("邊寬不能是負的，灰階要在 0 到 255");
        int width = image.Width + border * 2;
        int height = image.Height + border * 2;
        var destination = new byte[width * height * image.Channels];
        if (!NativeCore.TryMakeBorder(image.Pixels, destination, image.Width, image.Height, image.Channels, border, gray))
        {
            for (int row = 0; row < height; row++)
            {
                for (int column = 0; column < width; column++)
                {
                    bool inside = row >= border && column >= border && row < border + image.Height && column < border + image.Width;
                    for (int channel = 0; channel < image.Channels; channel++)
                    {
                        byte value = (byte)gray;
                        if (inside)
                            value = image.Pixels[((row - border) * image.Width + (column - border)) * image.Channels + channel];
                        destination[(row * width + column) * image.Channels + channel] = value;
                    }
                }
            }
        }
        return ImageBuffer.Wrap(width, height, image.Channels, destination, null);
    }

    public static ImageBuffer GaussPyramid(ImageBuffer image, string mode, double scale)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (image.Width < 2 || image.Height < 2)
            throw new ArgumentException("影像至少要 2×2");
        if (Math.Abs(scale - 0.5) > 1e-9)
            throw new ArgumentException("這版的金字塔只縮成一半");
        int kind = PyramidMode(mode);
        int width = kind >= 2 ? image.Width / 2 : (image.Width + 1) / 2;
        int height = kind >= 2 ? image.Height / 2 : (image.Height + 1) / 2;
        var destination = new byte[width * height * image.Channels];
        if (!NativeCore.TryPyramid(image.Pixels, destination, image.Width, image.Height, image.Channels, kind))
            PyramidScalar(image, destination, width, height, kind);
        return ImageBuffer.Wrap(width, height, image.Channels, destination, null);
    }

    public static ImageBuffer EntropyImage(ImageBuffer image, int maskWidth, int maskHeight)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (maskWidth < 1 || maskHeight < 1 || (maskWidth & 1) == 0 || (maskHeight & 1) == 0)
            throw new ArgumentException("視窗的寬和高要是正的奇數");
        return CompassGray(image, (source, destination) =>
        {
            if (NativeCore.TryEntropy(source, destination, image.Width, image.Height, maskWidth, maskHeight))
                return;
            EntropyScalar(source, destination, image.Width, image.Height, maskWidth, maskHeight);
        });
    }

    static ImageBuffer InverseWave(ImageBuffer image, int op)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryWave(source, destination, source.Length, op))
                return;
            for (int index = 0; index < source.Length; index++)
            {
                double sample = source[index];
                double waved;
                if (op == 5)
                    waved = Math.Atan((sample - 128.0) / 32.0);
                else
                {
                    double unit = (sample - 128.0) / 127.0;
                    if (unit < -1.0)
                        unit = -1.0;
                    if (unit > 1.0)
                        unit = 1.0;
                    waved = op == 3 ? Math.Asin(unit) : Math.Acos(unit);
                }
                if (op == 4)
                    waved = waved / Math.PI * 255.0;
                else
                    waved = (waved + Math.PI / 2.0) / Math.PI * 255.0;
                destination[index] = RoundD(waved);
            }
        });
    }

    static bool TileVertical(string mode)
    {
        string name = (mode ?? "").Trim().ToLowerInvariant();
        if (name == "horizontal")
            return false;
        if (name == "vertical")
            return true;
        throw new ArgumentException("Mode 只能是 horizontal 或 vertical");
    }

    static void TileScalar(ImageBuffer left, ImageBuffer right, byte[] destination, int width, bool vertical)
    {
        Array.Clear(destination, 0, destination.Length);
        CopyTile(left.Pixels, left.Width, left.Height, destination, width, 0, 0, left.Channels);
        CopyTile(right.Pixels, right.Width, right.Height, destination, width, vertical ? 0 : left.Width, vertical ? left.Height : 0, left.Channels);
    }

    static void CopyTile(byte[] source, int sourceWidth, int sourceHeight, byte[] destination, int destWidth, int originColumn, int originRow, int channels)
    {
        for (int row = 0; row < sourceHeight; row++)
        {
            for (int column = 0; column < sourceWidth; column++)
            {
                for (int channel = 0; channel < channels; channel++)
                    destination[((originRow + row) * destWidth + originColumn + column) * channels + channel] =
                        source[(row * sourceWidth + column) * channels + channel];
            }
        }
    }

    static int ConvolutionBorder(string margin)
    {
        string name = (margin ?? "").Trim().ToLowerInvariant();
        if (name == "continued")
            return 0;
        if (name == "cyclic")
            return 1;
        if (name == "mirror")
            return 2;
        throw new ArgumentException("Margin 只能是 continued、cyclic 或 mirror");
    }

    static float[] ConvolutionKernel(string? preset, double[]? weights, int maskWidth, out int kernelWidth, out int kernelHeight)
    {
        if (weights != null)
        {
            if (maskWidth < 1 || weights.Length % maskWidth != 0)
                throw new ArgumentException("MaskWidth 要能整除係數個數");
            kernelWidth = maskWidth;
            kernelHeight = weights.Length / maskWidth;
            if (kernelWidth > 15 || kernelHeight > 15)
                throw new ArgumentException("濾鏡最大是 15");
            var kernel = new float[weights.Length];
            for (int index = 0; index < weights.Length; index++)
                kernel[index] = (float)weights[index];
            return kernel;
        }
        string name = (preset ?? "").Trim().ToLowerInvariant();
        kernelWidth = 3;
        kernelHeight = 3;
        return name switch
        {
            "sobel_x" => new float[] { -1, 0, 1, -2, 0, 2, -1, 0, 1 },
            "sobel_y" => new float[] { -1, -2, -1, 0, 0, 0, 1, 2, 1 },
            "laplace" => new float[] { 0, -1, 0, -1, 4, -1, 0, -1, 0 },
            "smooth3" => new float[] { 1f / 9f, 1f / 9f, 1f / 9f, 1f / 9f, 1f / 9f, 1f / 9f, 1f / 9f, 1f / 9f, 1f / 9f },
            "sharpen" => new float[] { 0, -1, 0, -1, 5, -1, 0, -1, 0 },
            _ => throw new ArgumentException("不認識的濾鏡。可用 sobel_x、sobel_y、laplace、smooth3、sharpen，或給一串數字"),
        };
    }

    static void ConvolScalar(byte[] source, byte[] destination, int width, int height, float[] kernel, int kernelHeight, int kernelWidth, int border)
    {
        double sum = 0;
        for (int index = 0; index < kernel.Length; index++)
            sum += kernel[index];
        bool offset = sum > -1e-6 && sum < 1e-6;
        int anchorY = kernelHeight / 2;
        int anchorX = kernelWidth / 2;
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                float response = 0f;
                for (int dy = 0; dy < kernelHeight; dy++)
                {
                    int yy = BorderIndex(row + dy - anchorY, height, border);
                    for (int dx = 0; dx < kernelWidth; dx++)
                    {
                        int xx = BorderIndex(column + dx - anchorX, width, border);
                        response += source[yy * width + xx] * kernel[dy * kernelWidth + dx];
                    }
                }
                if (offset)
                    response += 128f;
                destination[row * width + column] = RoundF(response);
            }
        }
    }

    static int BorderIndex(int position, int length, int border)
    {
        if (border == 1)
        {
            int wrapped = position % length;
            if (wrapped < 0)
                wrapped += length;
            return wrapped;
        }
        if (border == 2)
            return FilterReflect(position, length);
        if (position < 0)
            return 0;
        if (position >= length)
            return length - 1;
        return position;
    }

    static int PyramidMode(string mode)
    {
        string name = (mode ?? "").Trim().ToLowerInvariant();
        if (name == "constant")
            return 0;
        if (name == "none")
            return 1;
        if (name == "min")
            return 2;
        if (name == "max")
            return 3;
        throw new ArgumentException("Mode 只能是 constant、none、min 或 max");
    }

    static void PyramidScalar(ImageBuffer image, byte[] destination, int outWidth, int outHeight, int mode)
    {
        if (mode == 0)
        {
            for (int channel = 0; channel < image.Channels; channel++)
            {
                var plane = new byte[image.Width * image.Height];
                for (int index = 0; index < plane.Length; index++)
                    plane[index] = image.Pixels[index * image.Channels + channel];
                byte[] reduced = PyrDownScalar(plane, image.Width, image.Height);
                for (int index = 0; index < reduced.Length; index++)
                    destination[index * image.Channels + channel] = reduced[index];
            }
            return;
        }
        for (int row = 0; row < outHeight; row++)
        {
            for (int column = 0; column < outWidth; column++)
            {
                for (int channel = 0; channel < image.Channels; channel++)
                {
                    byte value;
                    if (mode == 1)
                        value = image.Pixels[(row * 2 * image.Width + column * 2) * image.Channels + channel];
                    else
                    {
                        int r0 = row * 2;
                        int c0 = column * 2;
                        value = image.Pixels[(r0 * image.Width + c0) * image.Channels + channel];
                        byte[] samples =
                        {
                            value,
                            image.Pixels[(r0 * image.Width + c0 + 1) * image.Channels + channel],
                            image.Pixels[((r0 + 1) * image.Width + c0) * image.Channels + channel],
                            image.Pixels[((r0 + 1) * image.Width + c0 + 1) * image.Channels + channel],
                        };
                        for (int index = 1; index < 4; index++)
                        {
                            if (mode == 2 && samples[index] < value)
                                value = samples[index];
                            if (mode == 3 && samples[index] > value)
                                value = samples[index];
                        }
                    }
                    destination[(row * outWidth + column) * image.Channels + channel] = value;
                }
            }
        }
    }

    static int PyrReflect(int position, int length)
    {
        if ((uint)position < (uint)length)
            return position;
        if (length <= 1)
            return 0;
        do
        {
            if (position < 0)
                position = -position - 1;
            else
                position = length - 1 - (position - length);
        }
        while ((uint)position >= (uint)length);
        return position;
    }

    static byte[] PyrDownScalar(byte[] source, int width, int height)
    {
        int[] kernel = { 1, 4, 6, 4, 1 };
        var horizontal = new int[width * height];
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                int sum = 0;
                for (int tap = -2; tap <= 2; tap++)
                    sum += kernel[tap + 2] * source[row * width + FilterReflect(column + tap, width)];
                horizontal[row * width + column] = sum;
            }
        }
        int outWidth = (width + 1) / 2;
        int outHeight = (height + 1) / 2;
        var destination = new byte[outWidth * outHeight];
        for (int row = 0; row < outHeight; row++)
        {
            for (int column = 0; column < outWidth; column++)
            {
                int sum = 0;
                for (int tap = -2; tap <= 2; tap++)
                {
                    int yy = FilterReflect(row * 2 + tap, height);
                    sum += kernel[tap + 2] * horizontal[yy * width + column * 2];
                }
                sum = (sum + 128) / 256;
                if (sum < 0)
                    sum = 0;
                if (sum > 255)
                    sum = 255;
                destination[row * outWidth + column] = (byte)sum;
            }
        }
        return destination;
    }

    static void EntropyScalar(byte[] source, byte[] destination, int width, int height, int maskWidth, int maskHeight)
    {
        int anchorX = maskWidth / 2;
        int anchorY = maskHeight / 2;
        int area = maskWidth * maskHeight;
        var hist = new int[256];
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                Array.Clear(hist, 0, 256);
                for (int dy = 0; dy < maskHeight; dy++)
                {
                    int yy = FilterReflect(row + dy - anchorY, height);
                    for (int dx = 0; dx < maskWidth; dx++)
                    {
                        int xx = FilterReflect(column + dx - anchorX, width);
                        hist[source[yy * width + xx]]++;
                    }
                }
                double entropy = 0;
                for (int value = 0; value < 256; value++)
                {
                    if (hist[value] == 0)
                        continue;
                    double probability = hist[value] / (double)area;
                    entropy -= probability * (Math.Log(probability) / Math.Log(2.0));
                }
                destination[row * width + column] = RoundD(entropy / 8.0 * 255.0);
            }
        }
    }
}
