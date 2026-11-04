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

    /// <summary>八個方向取最大，再除以 15。垂直或水平階梯的強度等於階梯高度，平坦處是 0。彩色先轉成亮度。</summary>
    /// <remarks>Kirsch 邊緣強度</remarks>
    /// <example>
    /// <code>Vizion.kirsch_amp (Image, ImageEdgeAmp)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <returns>Kirsch 邊緣強度。彩色先轉成亮度。</returns>
    public static ImageBuffer KirschAmp(ImageBuffer image) =>
        CompassGray(image, (source, destination) =>
        {
            if (NativeCore.TryKirsch(source, destination, image.Width, image.Height))
                return;
            MaxKernel(source, destination, image.Width, image.Height, KirschKernels, 15.0, absolute: false);
        });

    /// <summary>四個邊緣遮罩的能量。平坦是 0。高度 40 的垂直階梯是 49。彩色先轉成亮度。</summary>
    /// <remarks>Frei-Chen 邊緣強度</remarks>
    /// <example>
    /// <code>Vizion.frei_amp (Image, ImageEdgeAmp)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <returns>Frei-Chen 邊緣強度。彩色先轉成亮度。</returns>
    public static ImageBuffer FreiAmp(ImageBuffer image) =>
        CompassGray(image, (source, destination) =>
        {
            if (NativeCore.TryFrei(source, destination, image.Width, image.Height))
                return;
            FreiScalar(source, destination, image.Width, image.Height);
        });

    /// <summary>八個方向取絕對值最大，不另外縮小。高度 40 的垂直階梯是 160。超過 255 會飽和。彩色先轉成亮度。</summary>
    /// <remarks>Robinson 邊緣強度</remarks>
    /// <example>
    /// <code>Vizion.robinson_amp (Image, ImageEdgeAmp)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <returns>Robinson 邊緣強度。彩色先轉成亮度。</returns>
    public static ImageBuffer RobinsonAmp(ImageBuffer image) =>
        CompassGray(image, (source, destination) =>
        {
            if (NativeCore.TryRobinsonAmp(source, destination, image.Width, image.Height))
                return;
            MaxKernel(source, destination, image.Width, image.Height, RobinsonKernels, 1.0, absolute: true);
        });

    /// <summary>和 sobel_dir 同一套：朝右是 128，正角度順時針。平坦時沒有方向，也記成朝右。要搭配 robinson_amp 才知道那裡有沒有邊緣。彩色先轉成亮度。</summary>
    /// <remarks>Robinson 邊緣方向</remarks>
    /// <example>
    /// <code>Vizion.robinson_dir (Image, ImageDir)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <returns>邊緣方向。朝右是 128。平坦也記成朝右。</returns>
    public static ImageBuffer RobinsonDir(ImageBuffer image) =>
        CompassGray(image, (source, destination) =>
        {
            if (NativeCore.TryRobinsonDir(source, destination, image.Width, image.Height))
                return;
            RobinsonDirScalar(source, destination, image.Width, image.Height);
        });

    /// <summary>128 是朝右，和 sobel_dir 一樣。正角度在畫面上順時針，朝下約是 191，朝左是 255。兩個方向都是 128 時記成朝右。彩色先轉成亮度。</summary>
    /// <remarks>兩個方向圖的角度</remarks>
    /// <example>
    /// <code>Vizion.atan2_image (ImageY, ImageX, ImageDir)</code>
    /// </example>
    /// <param name="vertical">列方向，128 是 0。</param>
    /// <param name="horizontal">行方向，128 是 0。</param>
    /// <returns>方向影像。定義域來自列方向那張。</returns>
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

    /// <summary>每個像素取視窗裡的最大值和最小值，再平均。平坦處不會變。彩色先轉成亮度。</summary>
    /// <remarks>局部最大與最小的平均</remarks>
    /// <example>
    /// <code>Vizion.midrange_image (Image, ImageMidrange, 3, 3)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="maskWidth">視窗寬，奇數。</param>
    /// <param name="maskHeight">視窗高，奇數。</param>
    /// <returns>局部最大和最小的平均。</returns>
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

    /// <summary>每個灰階直接當弧度。正弦是 -1 到 1，拉成 0 到 255，0 對應 128。灰階 0 的結果是 128。彩色每個通道分開算。</summary>
    /// <remarks>灰階取正弦</remarks>
    /// <example>
    /// <code>Vizion.sin_image (Image, ImageSin)</code>
    /// </example>
    /// <param name="image">灰階當弧度。</param>
    /// <returns>正弦拉回 0 到 255。灰階 0 的結果是 128。</returns>
    public static ImageBuffer SinImage(ImageBuffer image) => Wave(image, 0);

    /// <summary>每個灰階直接當弧度。餘弦是 -1 到 1，拉成 0 到 255，0 對應 128。灰階 0 的結果是 255。彩色每個通道分開算。</summary>
    /// <remarks>灰階取餘弦</remarks>
    /// <example>
    /// <code>Vizion.cos_image (Image, ImageCos)</code>
    /// </example>
    /// <param name="image">灰階當弧度。</param>
    /// <returns>餘弦拉回 0 到 255。灰階 0 的結果是 255。</returns>
    public static ImageBuffer CosImage(ImageBuffer image) => Wave(image, 1);

    /// <summary>每個灰階直接當弧度。正切先夾在 -16 到 16，再拉成 0 到 255，0 對應 128。灰階 0 的結果是 128。彩色每個通道分開算。</summary>
    /// <remarks>灰階取正切</remarks>
    /// <example>
    /// <code>Vizion.tan_image (Image, ImageTan)</code>
    /// </example>
    /// <param name="image">灰階當弧度。</param>
    /// <returns>正切先夾在 -16 到 16，再拉回 0 到 255。</returns>
    public static ImageBuffer TanImage(ImageBuffer image) => Wave(image, 2);

    /// <summary>mean 是視窗平均。gauss 用同一個奇數視窗。Alpha 至少是 1 的奇數。彩色每個通道分開做。</summary>
    /// <remarks>用均值或高斯把影像抹平</remarks>
    /// <example>
    /// <code>Vizion.smooth_image (Image, ImageSmooth, 'mean', 3)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="filter">mean 或 gauss。</param>
    /// <param name="alpha">奇數視窗。</param>
    /// <returns>平滑後的影像。定義域沿用輸入。</returns>
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
