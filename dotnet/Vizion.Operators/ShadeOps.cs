namespace Vizion;

public static partial class Operators
{
    /// <summary>細的高斯減去粗的高斯，得到中間那個尺度的邊緣，再加回原圖。比細尺度更碎的雜訊不會被放大。彩色只調亮度。</summary>
    /// <remarks>只加強某個粗細的邊緣。粗的尺度必須大於細的。</remarks>
    /// <example>
    /// <code>Vizion.edge_band (Image, Enhanced, 0.8, 2.4, 1, 2)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="sigmaFine">比這個更細的起伏不加。</param>
    /// <param name="sigmaCoarse">比這個更寬的明暗不加。</param>
    /// <param name="factor">加強多少。0 是原圖。</param>
    /// <param name="threshold">比這個小的起伏不加。</param>
    /// <returns>邊緣加強後的影像。定義域外的像素保持原值。</returns>
    public static ImageBuffer EdgeBand(ImageBuffer image, double sigmaFine, double sigmaCoarse, double factor, double threshold)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (sigmaFine < 0)
            throw new ArgumentException("SigmaFine 不能是負的");
        if (sigmaCoarse < 0)
            throw new ArgumentException("SigmaCoarse 不能是負的");
        if (!(sigmaCoarse > sigmaFine))
            throw new ArgumentException("粗的尺度要大於細的");
        if (factor < 0)
            throw new ArgumentException("Factor 不能是負的");
        if (threshold < 0)
            throw new ArgumentException("Threshold 不能是負的");
        byte[]? domain = OwnDomain(image);
        var destination = new byte[image.Pixels.Length];
        if (!NativeCore.TryEdgeBand(image.Pixels, destination, image.Width, image.Height, image.Channels, sigmaFine, sigmaCoarse, factor, threshold, domain))
            EdgeBandScalar(image.Pixels, destination, image.Width, image.Height, image.Channels, sigmaFine, sigmaCoarse, factor, threshold, domain);
        return ImageBuffer.Wrap(image.Width, image.Height, image.Channels, destination, image.Domain);
    }

    /// <summary>和 edge_enhance 一樣把原圖減模糊的差加回去，但先乘上梯度的比例。平坦的地方比例是 0。彩色只調亮度。</summary>
    /// <remarks>只在梯度大的地方加強。梯度達到 Level 才全額加強。</remarks>
    /// <example>
    /// <code>Vizion.edge_masked (Image, Enhanced, 1.2, 1, 30)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="sigma">邊緣大概多寬。0 是不處理。</param>
    /// <param name="factor">加強多少。0 是原圖。</param>
    /// <param name="level">梯度達到這個值就全額加強。</param>
    /// <returns>只在邊緣上加強的影像。定義域外的像素保持原值。</returns>
    public static ImageBuffer EdgeMasked(ImageBuffer image, double sigma, double factor, double level)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (sigma < 0)
            throw new ArgumentException("Sigma 不能是負的");
        if (factor < 0)
            throw new ArgumentException("Factor 不能是負的");
        if (level < 0)
            throw new ArgumentException("Level 不能是負的");
        if (!(level > 0))
            throw new ArgumentException("Level 要大於 0");
        byte[]? domain = OwnDomain(image);
        var destination = new byte[image.Pixels.Length];
        if (!NativeCore.TryEdgeMasked(image.Pixels, destination, image.Width, image.Height, image.Channels, sigma, factor, level, domain))
            EdgeMaskedScalar(image.Pixels, destination, image.Width, image.Height, image.Channels, sigma, factor, level, domain);
        return ImageBuffer.Wrap(image.Width, image.Height, image.Channels, destination, image.Domain);
    }

    /// <summary>梯度小的地方改用高斯模糊，梯度達到 Level 的地方保持原圖。彩色只調亮度。</summary>
    /// <remarks>平坦處平滑，邊緣留著。門檻是梯度，不是灰階差。</remarks>
    /// <example>
    /// <code>Vizion.edge_preserve (Image, Smoothed, 1.2, 30)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="sigma">平坦處模糊多寬。</param>
    /// <param name="level">梯度達到這個值就保留原圖。</param>
    /// <returns>平坦處變平滑的影像。定義域外的像素保持原值。</returns>
    public static ImageBuffer EdgePreserve(ImageBuffer image, double sigma, double level)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (sigma < 0)
            throw new ArgumentException("Sigma 不能是負的");
        if (level < 0)
            throw new ArgumentException("Level 不能是負的");
        if (!(level > 0))
            throw new ArgumentException("Level 要大於 0");
        byte[]? domain = OwnDomain(image);
        var destination = new byte[image.Pixels.Length];
        if (!NativeCore.TryEdgePreserve(image.Pixels, destination, image.Width, image.Height, image.Channels, sigma, level, domain))
            EdgePreserveScalar(image.Pixels, destination, image.Width, image.Height, image.Channels, sigma, level, domain);
        return ImageBuffer.Wrap(image.Width, image.Height, image.Channels, destination, image.Domain);
    }

    /// <summary>四鄰域的 Perona-Malik。parabolic 用 1/(1+(梯度/Contrast)²)。最外一圈像素保持不動。彩色先轉成亮度。</summary>
    /// <remarks>保邊的擴散平滑。</remarks>
    /// <example>
    /// <code>Vizion.anisotropic_diffusion (Image, ImageAniso, 'perona-malik', 20, 0.2, 10)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="mode">perona-malik 或 parabolic。</param>
    /// <param name="contrast">對比尺度。邊緣比這個陡就比較不會被抹掉。</param>
    /// <param name="theta">每步的時間，最大 0.25。</param>
    /// <param name="iterations">次數，1 到 40。</param>
    /// <returns>擴散後的灰階影像。定義域沿用輸入。</returns>
    public static ImageBuffer Anisotropic(ImageBuffer image, string mode, double contrast, double theta, int iterations)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        int kind = DiffusionCode(mode);
        if (contrast <= 0)
            throw new ArgumentException("Contrast 必須大於 0");
        if (theta <= 0 || theta > 0.25)
            throw new ArgumentException("Theta 要大於 0，而且不能超過 0.25");
        if (iterations < 1 || iterations > 40)
            throw new ArgumentException("Iterations 要在 1 到 40");
        var destination = new byte[image.Width * image.Height];
        if (!NativeCore.TryAnisotropic(image.Pixels, destination, image.Width, image.Height, image.Channels, kind, contrast, theta, iterations))
            AnisotropicScalar(image.Pixels, destination, image.Width, image.Height, image.Channels, kind, contrast, theta, iterations);
        return ImageBuffer.Wrap(image.Width, image.Height, 1, destination, image.Domain);
    }

    /// <summary>亮區的拉普拉斯是負的，會再變亮；暗區再變暗。canny 改看梯度方向的二階導數。完全平的地方不會變。彩色先轉成亮度。</summary>
    /// <remarks>把邊緣變陡。</remarks>
    /// <example>
    /// <code>Vizion.shock_filter (Image, Sharpened, 0.5, 5, 'laplace', 0.8)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="theta">每步的強度，最大 1。</param>
    /// <param name="iterations">次數，1 到 20。</param>
    /// <param name="mode">laplace 或 canny。</param>
    /// <param name="sigma">先平滑的尺度。0 表示不再平滑。</param>
    /// <returns>變陡之後的灰階影像。定義域沿用輸入。</returns>
    public static ImageBuffer ShockFilter(ImageBuffer image, double theta, int iterations, string mode, double sigma)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (theta <= 0 || theta > 1)
            throw new ArgumentException("Theta 要大於 0，而且不能超過 1");
        if (iterations < 1 || iterations > 20)
            throw new ArgumentException("Iterations 要在 1 到 20");
        int kind = ShockCode(mode);
        if (sigma < 0)
            throw new ArgumentException("Sigma 不能是負的");
        var destination = new byte[image.Width * image.Height];
        if (!NativeCore.TryShock(image.Pixels, destination, image.Width, image.Height, image.Channels, theta, iterations, kind, sigma))
            ShockScalar(image.Pixels, destination, image.Width, image.Height, image.Channels, theta, iterations, kind, sigma);
        return ImageBuffer.Wrap(image.Width, image.Height, 1, destination, image.Domain);
    }

    /// <summary>He、Sun、Tang 2010：用影像自己當引導。平坦處的係數接近 0，輸出變成局部平均；階梯邊緣的變異大，係數接近 1。彩色用亮度當引導。</summary>
    /// <remarks>引導濾波。Eps 是加在變異上的常數。</remarks>
    /// <example>
    /// <code>Vizion.guided_filter (Image, Filtered, 8, 800)</code>
    /// </example>
    /// <param name="image">輸入影像，同時當引導圖。</param>
    /// <param name="radius">半徑，像素。</param>
    /// <param name="eps">平滑強度，灰階平方。</param>
    /// <returns>保留邊緣的平滑結果。定義域沿用輸入。</returns>
    public static ImageBuffer GuidedFilter(ImageBuffer image, int radius, double eps)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (radius < 1)
            throw new ArgumentException("Radius 至少要 1");
        if (eps <= 0)
            throw new ArgumentException("Eps 要大於 0");
        var destination = new byte[image.Pixels.Length];
        if (!NativeCore.TryGuided(image.Pixels, destination, image.Width, image.Height, image.Channels, radius, eps))
            GuidedScalar(image.Pixels, destination, image.Width, image.Height, image.Channels, radius, eps);
        return ImageBuffer.Wrap(image.Width, image.Height, image.Channels, destination, image.Domain);
    }

    /// <summary>1998 年 Frangi 用 Hessian 的兩個特徵值分辨細線和方塊。細線接近 255，方塊內部和平坦處接近 0。彩色先轉成亮度。</summary>
    /// <remarks>用 Frangi 把細線加強。SigmaMax 比 Sigma 大時，中間再取一層，留下最亮的那層。</remarks>
    /// <example>
    /// <code>Vizion.frangi (Image, Response, 1.2, 1.2, 'dark')</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="sigma">最小尺度，像素。</param>
    /// <param name="sigmaMax">最大尺度。和 Sigma 相同就只做一層。</param>
    /// <param name="lightDark">dark 暗線或 light 亮線。</param>
    /// <returns>細線加強後的灰階。定義域沿用輸入。</returns>
    public static ImageBuffer Frangi(ImageBuffer image, double sigma, double sigmaMax, string lightDark)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (sigma <= 0 || sigmaMax <= 0)
            throw new ArgumentException("Sigma 和 SigmaMax 必須大於 0");
        if (sigmaMax < sigma)
            throw new ArgumentException("SigmaMax 不能小於 Sigma");
        int dark = FrangiDark(lightDark);
        var destination = new byte[image.Width * image.Height];
        if (!NativeCore.TryFrangi(image.Pixels, destination, image.Width, image.Height, image.Channels, sigma, sigmaMax, dark))
            FrangiScalar(image.Pixels, destination, image.Width, image.Height, image.Channels, sigma, sigmaMax, dark);
        return ImageBuffer.Wrap(image.Width, image.Height, 1, destination, image.Domain);
    }

    /// <summary>Zuiderveld 1994：把影像切成格子，每一格各自做直方圖等化，再把相鄰格子的結果混起來。彩色影像只處理亮度。</summary>
    /// <remarks>局部直方圖。ClipLimit 限制單一灰階被放大的程度。</remarks>
    /// <example>
    /// <code>Vizion.emphasize_clahe (Image, Emphasized, 8, 8)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="clipLimit">對比上限，愈大愈用力。</param>
    /// <param name="tiles">每邊切幾塊。</param>
    /// <returns>局部拉開對比的影像。定義域沿用輸入。</returns>
    public static ImageBuffer EmphasizeClahe(ImageBuffer image, double clipLimit, int tiles)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (clipLimit <= 0)
            throw new ArgumentException("ClipLimit 要大於 0");
        if (tiles < 2)
            throw new ArgumentException("Tiles 至少要 2");
        var destination = new byte[image.Pixels.Length];
        if (!NativeCore.TryClahe(image.Pixels, destination, image.Width, image.Height, image.Channels, clipLimit, tiles))
            ClaheScalar(image.Pixels, destination, image.Width, image.Height, image.Channels, clipLimit, tiles);
        return ImageBuffer.Wrap(image.Width, image.Height, image.Channels, destination, image.Domain);
    }

    /// <summary>樣板的 (0, 0) 放在影像左上角，比較小時其餘補 0。分數是灰階乘積的和，畫出來時負分變成 0，超過 255 會飽和。彩色先轉成亮度。</summary>
    /// <remarks>把循環相關畫成灰階。要看真正的分數請用 locate_template。</remarks>
    /// <example>
    /// <code>Vizion.correlation_image (Image, Template, ImageScore)</code>
    /// </example>
    /// <param name="image">要搜尋的影像。</param>
    /// <param name="template">樣板，灰階就是權重。</param>
    /// <returns>相關圖。不保留定義域。</returns>
    public static ImageBuffer CorrelationImage(ImageBuffer image, ImageBuffer template)
    {
        return Circular(image, template, 1, "樣板不能比影像大");
    }

    /// <summary>核的 (0, 0) 放在影像左上角。核比較小時其餘補 0。超出影像的部分從對面捲回來。彩色先轉成亮度。</summary>
    /// <remarks>用傅立葉做循環卷積。權重不會再除以總和。</remarks>
    /// <example>
    /// <code>Vizion.convol_fft (Image, Kernel, ImageResult)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="kernel">卷積核。</param>
    /// <returns>卷積結果。不保留定義域。</returns>
    public static ImageBuffer ConvolFft(ImageBuffer image, ImageBuffer kernel)
    {
        return Circular(image, kernel, 0, "核不能比影像大");
    }

    static ImageBuffer Circular(ImageBuffer image, ImageBuffer kernel, int correlation, string tooBig)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (kernel == null)
            throw new ArgumentNullException(nameof(kernel));
        byte[] source = image.GrayBytes();
        byte[] weights = kernel.GrayBytes();
        if (kernel.Height > image.Height || kernel.Width > image.Width)
            throw new ArgumentException(tooBig);
        var destination = new byte[image.Width * image.Height];
        if (!NativeCore.TryCircular(source, weights, destination, image.Width, image.Height, kernel.Width, kernel.Height, correlation))
            CircularScalar(source, weights, destination, image.Width, image.Height, kernel.Width, kernel.Height, correlation);
        return ImageBuffer.Wrap(image.Width, image.Height, 1, destination, null);
    }

    static int DiffusionCode(string mode)
    {
        if (mode == null)
            throw new ArgumentNullException(nameof(mode));
        string name = mode.Trim().Trim('\'').Trim('"').ToLowerInvariant();
        if (name == "perona-malik")
            return 0;
        if (name == "parabolic")
            return 1;
        throw new ArgumentException("Mode 只能是 perona-malik 或 parabolic");
    }

    static int ShockCode(string mode)
    {
        if (mode == null)
            throw new ArgumentNullException(nameof(mode));
        string name = mode.Trim().Trim('\'').Trim('"').ToLowerInvariant();
        if (name == "laplace")
            return 0;
        if (name == "canny")
            return 1;
        throw new ArgumentException("Mode 只能是 laplace 或 canny");
    }

    static int FrangiDark(string lightDark)
    {
        if (lightDark == null)
            throw new ArgumentNullException(nameof(lightDark));
        string name = lightDark.Trim().Trim('\'').Trim('"').ToLowerInvariant();
        if (name == "dark")
            return 1;
        if (name == "light")
            return 0;
        throw new ArgumentException("LightDark 只能是 dark 或 light");
    }

    static void LoadLuma(byte[] source, float[] luma, int pixels, int channels)
    {
        if (channels == 1)
        {
            for (int i = 0; i < pixels; i++)
                luma[i] = source[i];
            return;
        }
        for (int i = 0; i < pixels; i++)
        {
            BgrToYCrCb(source[i * 3], source[i * 3 + 1], source[i * 3 + 2], out int y, out _, out _);
            luma[i] = y;
        }
    }

    static void BlurSigma(float[] source, float[] destination, int width, int height, double sigma)
    {
        if (!(sigma > 0))
        {
            Array.Copy(source, destination, source.Length);
            return;
        }
        int side = BlurSide(sigma);
        float[] kernel = GaussKernel(sigma, side);
        float[] blurred = GradeSep(source, width, height, kernel, kernel);
        Array.Copy(blurred, destination, destination.Length);
    }

    static void PaintFloat(byte[] source, byte[] destination, float[] luma, float[] add, int pixels, int channels, byte[]? domain)
    {
        if (channels == 1)
        {
            for (int i = 0; i < pixels; i++)
                destination[i] = RoundF(luma[i] + add[i]);
        }
        else
        {
            Buffer.BlockCopy(source, 0, destination, 0, source.Length);
            for (int i = 0; i < pixels; i++)
            {
                float boost = add[i];
                if (!(boost > 1e-6f || boost < -1e-6f))
                    continue;
                BgrToYCrCb(source[i * 3], source[i * 3 + 1], source[i * 3 + 2], out int y, out int cr, out int cb);
                YCrCbToBgr((int)RoundF(luma[i] + boost), cr, cb, destination, i * 3);
            }
        }
        RestoreDomain(source, destination, pixels, channels, domain);
    }

    static void PaintDouble(byte[] source, byte[] destination, float[] luma, double[] add, int pixels, int channels, byte[]? domain)
    {
        if (channels == 1)
        {
            for (int i = 0; i < pixels; i++)
                destination[i] = RoundD(luma[i] + add[i]);
        }
        else
        {
            Buffer.BlockCopy(source, 0, destination, 0, source.Length);
            for (int i = 0; i < pixels; i++)
            {
                if (!(add[i] > 1e-6 || add[i] < -1e-6))
                    continue;
                BgrToYCrCb(source[i * 3], source[i * 3 + 1], source[i * 3 + 2], out int y, out int cr, out int cb);
                YCrCbToBgr((int)RoundD(luma[i] + add[i]), cr, cb, destination, i * 3);
            }
        }
        RestoreDomain(source, destination, pixels, channels, domain);
    }

    static void RestoreDomain(byte[] source, byte[] destination, int pixels, int channels, byte[]? domain)
    {
        if (domain == null)
            return;
        for (int i = 0; i < pixels; i++)
        {
            if (domain[i] != 0)
                continue;
            for (int channel = 0; channel < channels; channel++)
                destination[i * channels + channel] = source[i * channels + channel];
        }
    }

    static float[] MagnitudeOf(float[] source, int width, int height)
    {
        float[] deriv = { -1f, 0f, 1f };
        float[] smooth = { 1f, 2f, 1f };
        float[] gx = GradeSep(source, width, height, deriv, smooth);
        float[] gy = GradeSep(source, width, height, smooth, deriv);
        var magnitude = new float[gx.Length];
        for (int i = 0; i < magnitude.Length; i++)
        {
            float square = gx[i] * gx[i] + gy[i] * gy[i];
            magnitude[i] = (float)Math.Sqrt(square);
        }
        return magnitude;
    }

    static void EdgeBandScalar(
        byte[] source, byte[] destination, int width, int height, int channels,
        double sigmaFine, double sigmaCoarse, double factor, double threshold, byte[]? domain)
    {
        int pixels = width * height;
        if (!(factor > 0))
        {
            Buffer.BlockCopy(source, 0, destination, 0, source.Length);
            return;
        }
        var luma = new float[pixels];
        var fine = new float[pixels];
        var coarse = new float[pixels];
        var add = new float[pixels];
        LoadLuma(source, luma, pixels, channels);
        BlurSigma(luma, fine, width, height, sigmaFine);
        BlurSigma(luma, coarse, width, height, sigmaCoarse);
        for (int i = 0; i < pixels; i++)
            add[i] = Boost(fine[i], coarse[i], factor, threshold, 0);
        PaintFloat(source, destination, luma, add, pixels, channels, domain);
    }

    static void EdgeMaskedScalar(
        byte[] source, byte[] destination, int width, int height, int channels,
        double sigma, double factor, double level, byte[]? domain)
    {
        int pixels = width * height;
        if (!(sigma > 0) || !(factor > 0))
        {
            Buffer.BlockCopy(source, 0, destination, 0, source.Length);
            return;
        }
        var luma = new float[pixels];
        var blurred = new float[pixels];
        LoadLuma(source, luma, pixels, channels);
        BlurSigma(luma, blurred, width, height, sigma);
        float[] magnitude = MagnitudeOf(blurred, width, height);
        var add = new double[pixels];
        for (int i = 0; i < pixels; i++)
        {
            double weight = magnitude[i] / level;
            double diff = luma[i] - (double)blurred[i];
            if (weight < 0)
                weight = 0;
            if (weight > 1)
                weight = 1;
            add[i] = diff * weight * factor;
        }
        PaintDouble(source, destination, luma, add, pixels, channels, domain);
    }

    static void EdgePreserveScalar(
        byte[] source, byte[] destination, int width, int height, int channels,
        double sigma, double level, byte[]? domain)
    {
        int pixels = width * height;
        if (!(sigma > 0))
        {
            Buffer.BlockCopy(source, 0, destination, 0, source.Length);
            return;
        }
        var luma = new float[pixels];
        var blurred = new float[pixels];
        LoadLuma(source, luma, pixels, channels);
        BlurSigma(luma, blurred, width, height, sigma);
        float[] magnitude = MagnitudeOf(blurred, width, height);
        var add = new double[pixels];
        for (int i = 0; i < pixels; i++)
        {
            double weight = magnitude[i] / level;
            double diff = luma[i] - (double)blurred[i];
            if (weight < 0)
                weight = 0;
            if (weight > 1)
                weight = 1;
            add[i] = diff * weight - diff;
        }
        PaintDouble(source, destination, luma, add, pixels, channels, domain);
    }

    static byte[] ToGray(byte[] source, int pixels, int channels)
    {
        if (channels == 1)
        {
            var copy = new byte[pixels];
            Buffer.BlockCopy(source, 0, copy, 0, pixels);
            return copy;
        }
        var gray = new byte[pixels];
        for (int i = 0; i < pixels; i++)
            gray[i] = (byte)GrayConvert.Bgr(source[i * 3], source[i * 3 + 1], source[i * 3 + 2]);
        return gray;
    }

    static void AnisotropicScalar(
        byte[] source, byte[] destination, int width, int height, int channels,
        int mode, double contrast, double theta, int iterations)
    {
        int pixels = width * height;
        byte[] gray = ToGray(source, pixels, channels);
        var work = new float[pixels];
        var next = new float[pixels];
        for (int i = 0; i < pixels; i++)
            work[i] = gray[i];
        double scale = contrast * contrast;
        for (int step = 0; step < iterations; step++)
        {
            Array.Copy(work, next, pixels);
            if (width < 3 || height < 3)
                break;
            for (int row = 1; row < height - 1; row++)
            {
                for (int col = 1; col < width - 1; col++)
                {
                    int at = row * width + col;
                    float center = work[at];
                    float north = work[at - width] - center;
                    float south = work[at + width] - center;
                    float west = work[at - 1] - center;
                    float east = work[at + 1] - center;
                    double nc;
                    double sc;
                    double wc;
                    double ec;
                    if (mode == 0)
                    {
                        nc = Math.Exp(-((double)(north * north)) / scale);
                        sc = Math.Exp(-((double)(south * south)) / scale);
                        wc = Math.Exp(-((double)(west * west)) / scale);
                        ec = Math.Exp(-((double)(east * east)) / scale);
                    }
                    else
                    {
                        nc = 1.0 / (1.0 + ((double)(north * north)) / scale);
                        sc = 1.0 / (1.0 + ((double)(south * south)) / scale);
                        wc = 1.0 / (1.0 + ((double)(west * west)) / scale);
                        ec = 1.0 / (1.0 + ((double)(east * east)) / scale);
                    }
                    double mix = ((nc * north + sc * south) + wc * west) + ec * east;
                    next[at] = (float)(center + theta * mix);
                }
            }
            Array.Copy(next, work, pixels);
        }
        for (int i = 0; i < pixels; i++)
            destination[i] = RoundF(work[i]);
    }

    static void ShockScalar(
        byte[] source, byte[] destination, int width, int height, int channels,
        double theta, int iterations, int mode, double sigma)
    {
        int pixels = width * height;
        byte[] gray = ToGray(source, pixels, channels);
        var work = new float[pixels];
        var smooth = new float[pixels];
        var gx = new float[pixels];
        var gy = new float[pixels];
        var guide = new float[pixels];
        for (int i = 0; i < pixels; i++)
            work[i] = gray[i];
        for (int step = 0; step < iterations; step++)
        {
            float[] used;
            if (!(sigma > 0))
                used = work;
            else
            {
                BlurSigma(work, smooth, width, height, sigma);
                used = smooth;
            }
            Array.Clear(gx, 0, pixels);
            Array.Clear(gy, 0, pixels);
            Array.Clear(guide, 0, pixels);
            if (width >= 3)
            {
                for (int row = 0; row < height; row++)
                {
                    for (int col = 1; col < width - 1; col++)
                    {
                        int at = row * width + col;
                        gx[at] = (float)(((double)used[at + 1] - used[at - 1]) * 0.5);
                    }
                }
            }
            if (height >= 3)
            {
                for (int row = 1; row < height - 1; row++)
                {
                    for (int col = 0; col < width; col++)
                    {
                        int at = row * width + col;
                        gy[at] = (float)(((double)used[at + width] - used[at - width]) * 0.5);
                    }
                }
            }
            if (width >= 3 && height >= 3 && mode == 0)
            {
                for (int row = 1; row < height - 1; row++)
                {
                    for (int col = 1; col < width - 1; col++)
                    {
                        int at = row * width + col;
                        float sum = ((used[at + 1] + used[at - 1]) + used[at + width]) + used[at - width];
                        guide[at] = (float)((double)sum - 4.0 * used[at]);
                    }
                }
            }
            else if (width >= 3 && height >= 3)
            {
                for (int row = 1; row < height - 1; row++)
                {
                    for (int col = 1; col < width - 1; col++)
                    {
                        int at = row * width + col;
                        float center = used[at];
                        float ix = gx[at];
                        float iy = gy[at];
                        float mag2 = ix * ix + iy * iy;
                        double gxx = ((double)used[at + 1] - 2.0 * center) + used[at - 1];
                        double gyy = ((double)used[at + width] - 2.0 * center) + used[at - width];
                        float inner = ((used[at + width + 1] - used[at + width - 1]) - used[at - width + 1]) + used[at - width - 1];
                        double gxy = 0.25 * inner;
                        double dir = 0;
                        if ((double)mag2 > 1e-6)
                        {
                            double num = (((double)(ix * ix)) * gxx + (2.0 * ix * iy * gxy)) + ((double)(iy * iy)) * gyy;
                            dir = num / mag2;
                        }
                        guide[at] = (float)dir;
                    }
                }
            }
            for (int i = 0; i < pixels; i++)
            {
                float square = gx[i] * gx[i] + gy[i] * gy[i];
                float mag = (float)Math.Sqrt(square);
                float sign = guide[i] > 0f ? 1f : guide[i] < 0f ? -1f : 0f;
                work[i] = (float)((double)work[i] - theta * sign * mag);
            }
        }
        for (int i = 0; i < pixels; i++)
            destination[i] = RoundF(work[i]);
    }

    static void Box32(float[] source, float[] destination, int width, int height, int radius)
    {
        int window = radius * 2 + 1;
        var horizontal = new double[width * height];
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                double acc = 0;
                for (int tap = -radius; tap <= radius; tap++)
                    acc += source[row * width + GradeReflect(col + tap, width)];
                horizontal[row * width + col] = acc / window;
            }
        }
        for (int col = 0; col < width; col++)
        {
            for (int row = 0; row < height; row++)
            {
                double acc = 0;
                for (int tap = -radius; tap <= radius; tap++)
                    acc += horizontal[GradeReflect(row + tap, height) * width + col];
                destination[row * width + col] = (float)(acc / window);
            }
        }
    }

    static void Box64(double[] source, double[] destination, int width, int height, int radius)
    {
        int window = radius * 2 + 1;
        var horizontal = new double[width * height];
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                double acc = 0;
                for (int tap = -radius; tap <= radius; tap++)
                    acc += source[row * width + GradeReflect(col + tap, width)];
                horizontal[row * width + col] = acc / window;
            }
        }
        for (int col = 0; col < width; col++)
        {
            for (int row = 0; row < height; row++)
            {
                double acc = 0;
                for (int tap = -radius; tap <= radius; tap++)
                    acc += horizontal[GradeReflect(row + tap, height) * width + col];
                destination[row * width + col] = acc / window;
            }
        }
    }

    static void GuidedScalar(byte[] source, byte[] destination, int width, int height, int channels, int radius, double eps)
    {
        int pixels = width * height;
        byte[] gray = ToGray(source, pixels, channels);
        var guide = new float[pixels];
        var meanG = new float[pixels];
        var boxGg = new float[pixels];
        var plane = new float[pixels];
        var meanS = new float[pixels];
        var product = new float[pixels];
        var coeff = new double[pixels];
        var offset = new double[pixels];
        var boxC = new double[pixels];
        var boxO = new double[pixels];
        for (int i = 0; i < pixels; i++)
        {
            guide[i] = gray[i];
            product[i] = guide[i] * guide[i];
        }
        Box32(guide, meanG, width, height, radius);
        Box32(product, boxGg, width, height, radius);
        for (int channel = 0; channel < channels; channel++)
        {
            for (int i = 0; i < pixels; i++)
                plane[i] = source[i * channels + channel];
            Box32(plane, meanS, width, height, radius);
            for (int i = 0; i < pixels; i++)
                product[i] = guide[i] * plane[i];
            Box32(product, plane, width, height, radius);
            for (int i = 0; i < pixels; i++)
            {
                float var32 = boxGg[i] - meanG[i] * meanG[i];
                double variance = var32;
                if (variance < 0)
                    variance = 0;
                float cov = plane[i] - meanG[i] * meanS[i];
                coeff[i] = cov / (variance + eps);
                offset[i] = meanS[i] - coeff[i] * meanG[i];
            }
            Box64(coeff, boxC, width, height, radius);
            Box64(offset, boxO, width, height, radius);
            for (int i = 0; i < pixels; i++)
                destination[i * channels + channel] = RoundD(boxC[i] * guide[i] + boxO[i]);
        }
    }

    static void DerivKernels(double sigma, out float[] smooth, out float[] first, out float[] second)
    {
        int radius = (int)Math.Ceiling(sigma * 4.0);
        if (radius < 1)
            radius = 1;
        int count = radius * 2 + 1;
        smooth = new float[count];
        first = new float[count];
        second = new float[count];
        double variance = sigma * sigma;
        var raw = new double[count];
        double sum = 0;
        for (int i = -radius; i <= radius; i++)
        {
            raw[i + radius] = Math.Exp(-0.5 * i * i / variance);
            sum += raw[i + radius];
        }
        for (int i = -radius; i <= radius; i++)
        {
            double weight = raw[i + radius] / sum;
            int at = i + radius;
            smooth[at] = (float)weight;
            first[at] = (float)((i / variance) * weight);
            second[at] = (float)(((i * (double)i / (variance * variance)) - (1.0 / variance)) * weight);
        }
    }

    static void FrangiScale(float[] gray, float[] vessel, int width, int height, double sigma, int dark)
    {
        int pixels = width * height;
        DerivKernels(sigma, out float[] smooth, out float[] first, out float[] second);
        float[] xx = GradeSep(gray, width, height, second, smooth);
        float[] yy = GradeSep(gray, width, height, smooth, second);
        float[] xy = GradeSep(gray, width, height, first, first);
        double scale = sigma * sigma;
        var ratio = new double[pixels];
        var magnitude = new double[pixels];
        var secondEigen = new double[pixels];
        double peak = 0;
        for (int i = 0; i < pixels; i++)
        {
            double dxx = xx[i] * scale;
            double dyy = yy[i] * scale;
            double dxy = xy[i] * scale;
            double trace = dxx + dyy;
            double det = dxx * dyy - dxy * dxy;
            double inside = trace * trace * 0.25 - det;
            if (inside < 0)
                inside = 0;
            double disc = Math.Sqrt(inside);
            double high = trace * 0.5 + disc;
            double low = trace * 0.5 - disc;
            bool swap = Math.Abs(high) < Math.Abs(low);
            double firstValue = swap ? high : low;
            double secondValue = swap ? low : high;
            double secondAbs = Math.Abs(secondValue);
            ratio[i] = secondAbs > 1e-8 ? Math.Abs(firstValue) / secondAbs : 0;
            magnitude[i] = Math.Sqrt(firstValue * firstValue + secondValue * secondValue);
            secondEigen[i] = secondValue;
            if (magnitude[i] > peak)
                peak = magnitude[i];
        }
        if (peak < 1e-8)
        {
            Array.Clear(vessel, 0, pixels);
            return;
        }
        double half = 0.5 * peak;
        for (int i = 0; i < pixels; i++)
        {
            bool keep = dark != 0 ? secondEigen[i] > 0 : secondEigen[i] < 0;
            double value = Math.Exp(-(ratio[i] * ratio[i]) / 0.5) * (1.0 - Math.Exp(-(magnitude[i] * magnitude[i]) / (2.0 * half * half)));
            vessel[i] = keep ? (float)value : 0f;
        }
    }

    static void FrangiScalar(
        byte[] source, byte[] destination, int width, int height, int channels,
        double sigma, double sigmaMax, int dark)
    {
        int pixels = width * height;
        byte[] bytes = ToGray(source, pixels, channels);
        var gray = new float[pixels];
        for (int i = 0; i < pixels; i++)
            gray[i] = bytes[i];
        var scales = new double[Math.Abs(sigmaMax - sigma) < 1e-6 ? 1 : 3];
        scales[0] = sigma;
        if (scales.Length == 3)
        {
            scales[1] = (sigma + sigmaMax) * 0.5;
            scales[2] = sigmaMax;
        }
        var vessel = new float[pixels];
        var best = new float[pixels];
        for (int scale = 0; scale < scales.Length; scale++)
        {
            FrangiScale(gray, vessel, width, height, scales[scale], dark);
            for (int i = 0; i < pixels; i++)
            {
                if (vessel[i] > best[i])
                    best[i] = vessel[i];
            }
        }
        for (int i = 0; i < pixels; i++)
            destination[i] = RoundD(best[i] * 255.0);
    }

    static void ClahePlane(byte[] source, byte[] destination, int width, int height, double clip, int tiles)
    {
        bool usePad = width % tiles != 0 || height % tiles != 0;
        int bottom = usePad ? tiles - (height % tiles) : 0;
        int right = usePad ? tiles - (width % tiles) : 0;
        int padWidth = width + right;
        int padHeight = height + bottom;
        int tileWidth = padWidth / tiles;
        int tileHeight = padHeight / tiles;
        int tileArea = tileWidth * tileHeight;
        float lutScale = 255f / tileArea;
        var padded = new byte[padWidth * padHeight];
        for (int y = 0; y < padHeight; y++)
        {
            int sy = y < height ? y : GradeReflect(y, height);
            for (int x = 0; x < padWidth; x++)
            {
                int sx = x < width ? x : GradeReflect(x, width);
                padded[y * padWidth + x] = source[sy * width + sx];
            }
        }
        int clipLimit = 0;
        if (clip > 0)
        {
            clipLimit = (int)(clip * tileArea / 256.0);
            if (clipLimit < 1)
                clipLimit = 1;
        }
        var lut = new byte[tiles * tiles * 256];
        var hist = new int[256];
        for (int ty = 0; ty < tiles; ty++)
        {
            for (int tx = 0; tx < tiles; tx++)
            {
                Array.Clear(hist, 0, 256);
                for (int y = 0; y < tileHeight; y++)
                {
                    int line = ((ty * tileHeight + y) * padWidth) + tx * tileWidth;
                    for (int x = 0; x < tileWidth; x++)
                        hist[padded[line + x]]++;
                }
                if (clipLimit > 0)
                {
                    int clipped = 0;
                    for (int bin = 0; bin < 256; bin++)
                    {
                        if (hist[bin] > clipLimit)
                        {
                            clipped += hist[bin] - clipLimit;
                            hist[bin] = clipLimit;
                        }
                    }
                    int redist = clipped / 256;
                    int residual = clipped - redist * 256;
                    for (int bin = 0; bin < 256; bin++)
                        hist[bin] += redist;
                    if (residual != 0)
                    {
                        int residualStep = 256 / residual;
                        if (residualStep < 1)
                            residualStep = 1;
                        for (int bin = 0; bin < 256 && residual > 0; bin += residualStep, residual--)
                            hist[bin]++;
                    }
                }
                int sum = 0;
                int row = (ty * tiles + tx) * 256;
                for (int bin = 0; bin < 256; bin++)
                {
                    sum += hist[bin];
                    lut[row + bin] = RoundF(sum * lutScale);
                }
            }
        }
        float invWidth = 1f / tileWidth;
        float invHeight = 1f / tileHeight;
        var along = new float[width];
        var alongOne = new float[width];
        var tx1s = new int[width];
        var tx2s = new int[width];
        for (int x = 0; x < width; x++)
        {
            float txf = x * invWidth - 0.5f;
            int tx1 = (int)Math.Floor(txf);
            int tx2 = tx1 + 1;
            along[x] = txf - tx1;
            alongOne[x] = 1f - along[x];
            if (tx1 < 0)
                tx1 = 0;
            if (tx2 > tiles - 1)
                tx2 = tiles - 1;
            tx1s[x] = tx1;
            tx2s[x] = tx2;
        }
        for (int y = 0; y < height; y++)
        {
            float tyf = y * invHeight - 0.5f;
            int ty1 = (int)Math.Floor(tyf);
            int ty2 = ty1 + 1;
            float ya = tyf - ty1;
            float ya1 = 1f - ya;
            if (ty1 < 0)
                ty1 = 0;
            if (ty2 > tiles - 1)
                ty2 = tiles - 1;
            for (int x = 0; x < width; x++)
            {
                int value = source[y * width + x];
                float a = lut[(ty1 * tiles + tx1s[x]) * 256 + value];
                float b = lut[(ty1 * tiles + tx2s[x]) * 256 + value];
                float c = lut[(ty2 * tiles + tx1s[x]) * 256 + value];
                float d = lut[(ty2 * tiles + tx2s[x]) * 256 + value];
                float res = ((a * alongOne[x] + b * along[x]) * ya1) + ((c * alongOne[x] + d * along[x]) * ya);
                destination[y * width + x] = RoundF(res);
            }
        }
    }

    static void ClaheScalar(byte[] source, byte[] destination, int width, int height, int channels, double clip, int tiles)
    {
        if (channels == 1)
        {
            ClahePlane(source, destination, width, height, clip, tiles);
            return;
        }
        int pixels = width * height;
        var plane = new byte[pixels];
        var chroma = new byte[pixels * 2];
        for (int i = 0; i < pixels; i++)
        {
            BgrToYCrCb(source[i * 3], source[i * 3 + 1], source[i * 3 + 2], out int y, out int cr, out int cb);
            plane[i] = (byte)y;
            chroma[i * 2] = (byte)cr;
            chroma[i * 2 + 1] = (byte)cb;
        }
        ClahePlane(plane, plane, width, height, clip, tiles);
        for (int i = 0; i < pixels; i++)
            YCrCbToBgr(plane[i], chroma[i * 2], chroma[i * 2 + 1], destination, i * 3);
    }

    static void CircularScalar(
        byte[] image, byte[] kernel, byte[] destination, int width, int height, int kernelWidth, int kernelHeight, int correlation)
    {
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                long acc = 0;
                for (int i = 0; i < kernelHeight; i++)
                {
                    int rr = correlation != 0 ? row + i : row - i;
                    if (rr >= height)
                        rr -= height;
                    if (rr < 0)
                        rr += height;
                    for (int j = 0; j < kernelWidth; j++)
                    {
                        int cc = correlation != 0 ? col + j : col - j;
                        if (cc >= width)
                            cc -= width;
                        if (cc < 0)
                            cc += width;
                        acc += (long)kernel[i * kernelWidth + j] * image[rr * width + cc];
                    }
                }
                if (acc <= 0)
                    destination[row * width + col] = 0;
                else if (acc >= 255)
                    destination[row * width + col] = 255;
                else
                    destination[row * width + col] = (byte)acc;
            }
        }
    }
}
