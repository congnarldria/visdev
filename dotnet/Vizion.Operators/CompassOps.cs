namespace Vizion;

public static partial class Operators
{
    static readonly float[][] KirschKernels =
    {
        new float[] { 5, 5, 5, -3, 0, -3, -3, -3, -3 },
        new float[] { 5, 5, -3, 5, 0, -3, -3, -3, -3 },
        new float[] { 5, -3, -3, 5, 0, -3, 5, -3, -3 },
        new float[] { -3, -3, -3, 5, 0, -3, 5, 5, -3 },
        new float[] { -3, -3, -3, -3, 0, -3, 5, 5, 5 },
        new float[] { -3, -3, -3, -3, 0, 5, -3, 5, 5 },
        new float[] { -3, -3, 5, -3, 0, 5, -3, -3, 5 },
        new float[] { -3, 5, 5, -3, 0, 5, -3, -3, -3 },
    };

    static readonly float[][] RobinsonKernels =
    {
        new float[] { -1, 0, 1, -2, 0, 2, -1, 0, 1 },
        new float[] { 0, 1, 2, -1, 0, 1, -2, -1, 0 },
        new float[] { 1, 2, 1, 0, 0, 0, -1, -2, -1 },
        new float[] { 2, 1, 0, 1, 0, -1, 0, -1, -2 },
        new float[] { 1, 0, -1, 2, 0, -2, 1, 0, -1 },
        new float[] { 0, -1, -2, 1, 0, -1, 2, 1, 0 },
        new float[] { -1, -2, -1, 0, 0, 0, 1, 2, 1 },
        new float[] { -2, -1, 0, -1, 0, 1, 0, 1, 2 },
    };

    public static ImageBuffer KirschAmp(ImageBuffer image) =>
        CompassGray(image, (source, destination) =>
        {
            if (NativeCore.TryKirsch(source, destination, image.Width, image.Height))
                return;
            MaxKernel(source, destination, image.Width, image.Height, KirschKernels, 15.0, absolute: false);
        });

    public static ImageBuffer FreiAmp(ImageBuffer image) =>
        CompassGray(image, (source, destination) =>
        {
            if (NativeCore.TryFrei(source, destination, image.Width, image.Height))
                return;
            FreiScalar(source, destination, image.Width, image.Height);
        });

    public static ImageBuffer RobinsonAmp(ImageBuffer image) =>
        CompassGray(image, (source, destination) =>
        {
            if (NativeCore.TryRobinsonAmp(source, destination, image.Width, image.Height))
                return;
            MaxKernel(source, destination, image.Width, image.Height, RobinsonKernels, 1.0, absolute: true);
        });

    public static ImageBuffer RobinsonDir(ImageBuffer image) =>
        CompassGray(image, (source, destination) =>
        {
            if (NativeCore.TryRobinsonDir(source, destination, image.Width, image.Height))
                return;
            RobinsonDirScalar(source, destination, image.Width, image.Height);
        });

    public static ImageBuffer Atan2Image(ImageBuffer vertical, ImageBuffer horizontal)
    {
        if (vertical == null || horizontal == null)
            throw new ArgumentNullException(nameof(vertical));
        if (vertical.Width != horizontal.Width || vertical.Height != horizontal.Height)
            throw new ArgumentException("兩張影像的尺寸不同");
        byte[] y = vertical.GrayBytes();
        byte[] x = horizontal.GrayBytes();
        var destination = new byte[y.Length];
        if (!NativeCore.TryAtan2Gray(y, x, destination, y.Length))
        {
            for (int index = 0; index < y.Length; index++)
            {
                double angle = Math.Atan2(y[index] - 128.0, x[index] - 128.0);
                destination[index] = RoundD((angle + Math.PI) / (2.0 * Math.PI) * 255.0);
            }
        }
        return ImageBuffer.Wrap(vertical.Width, vertical.Height, 1, destination, vertical.Domain);
    }

    public static ImageBuffer MidrangeImage(ImageBuffer image, int maskWidth, int maskHeight)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (maskWidth < 1 || maskHeight < 1 || (maskWidth & 1) == 0 || (maskHeight & 1) == 0)
            throw new ArgumentException("視窗的寬和高要是正的奇數");
        return CompassGray(image, (source, destination) =>
        {
            if (NativeCore.TryMidrange(source, destination, image.Width, image.Height, maskWidth, maskHeight))
                return;
            int anchorX = maskWidth / 2;
            int anchorY = maskHeight / 2;
            for (int row = 0; row < image.Height; row++)
            {
                for (int column = 0; column < image.Width; column++)
                {
                    int darkest = 255;
                    int brightest = 0;
                    for (int dy = 0; dy < maskHeight; dy++)
                    {
                        int yy = FilterReflect(row + dy - anchorY, image.Height);
                        for (int dx = 0; dx < maskWidth; dx++)
                        {
                            int xx = FilterReflect(column + dx - anchorX, image.Width);
                            int value = source[yy * image.Width + xx];
                            if (value < darkest)
                                darkest = value;
                            if (value > brightest)
                                brightest = value;
                        }
                    }
                    destination[row * image.Width + column] = RoundD((darkest + brightest) / 2.0);
                }
            }
        });
    }

    public static ImageBuffer SinImage(ImageBuffer image) => Wave(image, 0);

    public static ImageBuffer CosImage(ImageBuffer image) => Wave(image, 1);

    public static ImageBuffer TanImage(ImageBuffer image) => Wave(image, 2);

    public static ImageBuffer SmoothImage(ImageBuffer image, string filter, int alpha)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (alpha < 1 || (alpha & 1) == 0)
            throw new ArgumentException("Alpha 要是正的奇數");
        string kind = (filter ?? "").Trim().Trim('\'').Trim('"').ToLowerInvariant();
        ImageBuffer smoothed;
        if (kind == "mean")
            smoothed = MeanImage(image, alpha, alpha);
        else if (kind == "gauss")
            smoothed = alpha < 3 ? CopyImage(image) : GaussImage(image, alpha);
        else
            throw new ArgumentException("Filter 只能是 mean 或 gauss");
        return ImageBuffer.Wrap(image.Width, image.Height, image.Channels, smoothed.Pixels, image.Domain);
    }

    static ImageBuffer CopyImage(ImageBuffer image)
    {
        var copy = new byte[image.Pixels.Length];
        Buffer.BlockCopy(image.Pixels, 0, copy, 0, copy.Length);
        return ImageBuffer.Wrap(image.Width, image.Height, image.Channels, copy, image.Domain);
    }

    static ImageBuffer Wave(ImageBuffer image, int op)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryWave(source, destination, source.Length, op))
                return;
            for (int index = 0; index < source.Length; index++)
            {
                double angle = source[index];
                double waved;
                if (op == 0)
                    waved = (Math.Sin(angle) + 1.0) * 127.5;
                else if (op == 1)
                    waved = (Math.Cos(angle) + 1.0) * 127.5;
                else
                {
                    waved = Math.Tan(angle);
                    if (waved < -16.0)
                        waved = -16.0;
                    if (waved > 16.0)
                        waved = 16.0;
                    waved = (waved + 16.0) / 32.0 * 255.0;
                }
                destination[index] = RoundD(waved);
            }
        });
    }

    static ImageBuffer CompassGray(ImageBuffer image, Action<byte[], byte[]> filter)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        return GrayKept(image, filter);
    }

    static void MaxKernel(byte[] source, byte[] destination, int width, int height, float[][] kernels, double scale, bool absolute)
    {
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                float best = 0f;
                bool any = false;
                foreach (float[] kernel in kernels)
                {
                    float response = Kernel3(source, width, height, row, column, kernel);
                    if (absolute)
                        response = Math.Abs(response);
                    if (!any || response > best)
                    {
                        best = response;
                        any = true;
                    }
                }
                destination[row * width + column] = RoundD(best / scale);
            }
        }
    }

    static void FreiScalar(byte[] source, byte[] destination, int width, int height)
    {
        float root = (float)Math.Sqrt(2.0);
        float scale = (float)(1.0 / (2.0 * Math.Sqrt(2.0)));
        float[][] kernels =
        {
            Scaled(scale, 1, root, 1, 0, 0, 0, -1, -root, -1),
            Scaled(scale, 1, 0, -1, root, 0, -root, 1, 0, -1),
            Scaled(scale, 0, -1, root, 1, 0, -1, -root, 1, 0),
            Scaled(scale, root, -1, 0, -1, 0, 1, 0, 1, -root),
        };
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                float energy = 0f;
                foreach (float[] kernel in kernels)
                {
                    float response = Kernel3(source, width, height, row, column, kernel);
                    energy += response * response;
                }
                destination[row * width + column] = RoundD(Math.Sqrt(energy));
            }
        }
    }

    static float[] Scaled(float scale, params float[] values)
    {
        var kernel = new float[values.Length];
        for (int index = 0; index < values.Length; index++)
            kernel[index] = scale * values[index];
        return kernel;
    }

    static void RobinsonDirScalar(byte[] source, byte[] destination, int width, int height)
    {
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                int bestIndex = 0;
                float best = Kernel3(source, width, height, row, column, RobinsonKernels[0]);
                for (int mask = 1; mask < RobinsonKernels.Length; mask++)
                {
                    float response = Kernel3(source, width, height, row, column, RobinsonKernels[mask]);
                    if (response > best)
                    {
                        best = response;
                        bestIndex = mask;
                    }
                }
                double angle = bestIndex * (Math.PI / 4.0);
                if (best < 0f)
                    angle += Math.PI;
                angle = Math.Atan2(Math.Sin(angle), Math.Cos(angle));
                destination[row * width + column] = RoundD((angle + Math.PI) / (2.0 * Math.PI) * 255.0);
            }
        }
    }
}
