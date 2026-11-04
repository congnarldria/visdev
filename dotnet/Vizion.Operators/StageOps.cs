namespace Vizion;

public static partial class Operators
{
    /// <summary>灰階 = Alpha × (列 − Row) + Beta × (行 − Column) + Gamma，再夾到 0 到 255。這版只做 byte。</summary>
    /// <remarks>建立灰階斜坡</remarks>
    /// <example>
    /// <code>Vizion.gen_image_gray_ramp (Ramp, 0, 2, 10, 0, 0, 40, 30)</code>
    /// </example>
    /// <param name="alpha">列方向每像素增加的灰階。</param>
    /// <param name="beta">行方向每像素增加的灰階。</param>
    /// <param name="gamma">基準點的灰階。</param>
    /// <param name="row">基準列。</param>
    /// <param name="column">基準行。</param>
    /// <param name="width">寬度。</param>
    /// <param name="height">高度。</param>
    /// <returns>斜坡影像。沒有定義域。</returns>
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

    /// <summary>這版一次兩張，不是一個影像 tuple。尺寸不同時，不足的地方補 0。通道數要相同。</summary>
    /// <remarks>把兩張影像接起來</remarks>
    /// <example>
    /// <code>Vizion.tile_images (Image1, Image2, TiledImage, 'horizontal')</code>
    /// </example>
    /// <param name="left">第一張影像。</param>
    /// <param name="right">第二張影像。</param>
    /// <param name="mode">horizontal 左右接，vertical 上下接。</param>
    /// <returns>接起來的影像。不足的地方補 0。不保留定義域。</returns>
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

    /// <summary>名稱可用 sobel_x、sobel_y、laplace、smooth3、sharpen。係數總和是 0 時，結果會加 128，128 代表 0。其他係數直接裁到 0 到 255，不會自動除以總和。彩色先轉成亮度。</summary>
    /// <remarks>用自訂濾鏡做卷積</remarks>
    /// <example>
    /// <code>Vizion.convol_image (Image, ImageResult, 'smooth3', 0, 'continued')</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="preset">sobel_x、sobel_y、laplace、smooth3、sharpen。用係數陣列時可空。</param>
    /// <param name="weights">一排係數。用名稱時可空。</param>
    /// <param name="maskWidth">係數是數字時的寬度。用名稱時填 0。</param>
    /// <param name="margin">continued、cyclic 或 mirror。</param>
    /// <returns>卷積結果。彩色先轉成亮度。不保留定義域。</returns>
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

    /// <summary>128 當作 0，0 和 255 拉到 -1 和 1。反正弦從 -π/2 到 π/2，再拉成 0 到 255。輸入 128 的結果是 128。</summary>
    /// <remarks>灰階取反正弦</remarks>
    /// <example>
    /// <code>Vizion.asin_image (Image, ImageAsin)</code>
    /// </example>
    /// <param name="image">128 是 0，0 和 255 是 -1 和 1。</param>
    /// <returns>反正弦。輸入 128 的結果是 128。</returns>
    public static ImageBuffer AsinImage(ImageBuffer image) => InverseWave(image, 3);

    /// <summary>128 當作 0。反餘弦從 0 到 π，再拉成 0 到 255。輸入 128 的結果是 128，輸入 255 的結果是 0。</summary>
    /// <remarks>灰階取反餘弦</remarks>
    /// <example>
    /// <code>Vizion.acos_image (Image, ImageAcos)</code>
    /// </example>
    /// <param name="image">128 是 0，0 和 255 是 -1 和 1。</param>
    /// <returns>反餘弦。輸入 128 的結果是 128，輸入 255 的結果是 0。</returns>
    public static ImageBuffer AcosImage(ImageBuffer image) => InverseWave(image, 4);

    /// <summary>128 當作 0，先除以 32 再取反正切。結果從 -π/2 到 π/2，拉成 0 到 255。輸入 128 的結果是 128，160 是 191。</summary>
    /// <remarks>灰階取反正切</remarks>
    /// <example>
    /// <code>Vizion.atan_image (Image, ImageAtan)</code>
    /// </example>
    /// <param name="image">128 是 0。</param>
    /// <returns>反正切。輸入 128 的結果是 128。</returns>
    public static ImageBuffer AtanImage(ImageBuffer image) => InverseWave(image, 5);

    /// <summary>小遮罩的平均減去大遮罩的平均。128 代表沒有變化，比 128 大是中間比外圈亮。偶數會加 1，最大 31，而且 Inner 要比 Outer 小。彩色先轉成亮度。</summary>
    /// <remarks>留下兩個尺度之間的變化</remarks>
    /// <example>
    /// <code>Vizion.bandpass_image (Image, ImageBandpass, 3, 9)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="inner">較小的正方形邊長。</param>
    /// <param name="outer">較大的正方形邊長。</param>
    /// <returns>帶通結果。128 代表沒有變化。</returns>
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

    /// <summary>上下左右加同樣寬度。新的像素填 Grayval，原來的像素位置往內縮。寬高各增加兩倍的邊寬。</summary>
    /// <remarks>在影像四周加邊</remarks>
    /// <example>
    /// <code>Vizion.add_image_border (Image, ImageBorder, 2, 0)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="border">四邊各加幾個像素。</param>
    /// <param name="gray">邊的灰階。</param>
    /// <returns>加上邊之後的影像。不保留定義域。</returns>
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

    /// <summary>constant 用高斯再取樣。none 取偶數行列。min 和 max 看每個 2×2。這版只縮成一半，不保留定義域。結果會變成之後區域的參考尺寸。</summary>
    /// <remarks>做一層一半大小的金字塔</remarks>
    /// <example>
    /// <code>Vizion.gen_gauss_pyramid (Image, ImagePyramid, 'constant', 0.5)</code>
    /// </example>
    /// <param name="image">輸入影像。至少 2×2。</param>
    /// <param name="mode">constant 是高斯，none 是隔點取，min 和 max 是 2×2 的最小或最大。</param>
    /// <param name="scale">只能是 0.5。</param>
    /// <returns>縮成一半的影像。不保留定義域。</returns>
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

    /// <summary>視窗裡每個灰階的比例取以 2 為底的熵。全是同一個灰階時是 0。256 種灰階一樣多時是 255。彩色先轉成亮度。</summary>
    /// <remarks>算每個像素附近的熵</remarks>
    /// <example>
    /// <code>Vizion.entropy_image (Image, ImageEntropy, 3, 3)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="maskWidth">視窗寬，奇數。</param>
    /// <param name="maskHeight">視窗高，奇數。</param>
    /// <returns>局部熵。全是同一個灰階時是 0。</returns>
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
