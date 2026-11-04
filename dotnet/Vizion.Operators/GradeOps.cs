namespace Vizion;

public static partial class Operators
{
    /// <summary>x 沿行、y 沿列，用可分離的高斯導數，不是先模糊再做 Sobel。有號導數以 128 為 0。gradient 是振幅、不加 128。彩色先轉成亮度。</summary>
    /// <remarks>看某個方向的灰階變化。平坦或直線斜坡的二階導數是 128。</remarks>
    /// <example>
    /// <code>Vizion.derivate_gauss (Image, DerivGauss, 1.0, 'x')</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="sigma">高斯標準差，必須大於 0。</param>
    /// <param name="component">none、x、y、xx、yy、xy、gradient 或 laplace。</param>
    /// <returns>導數影像。不保留定義域。</returns>
    public static ImageBuffer DerivGauss(ImageBuffer image, double sigma, string component)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (!(sigma > 0))
            throw new ArgumentException("Sigma 必須大於 0");
        int kind = DerivCode(component);
        byte[] gray = image.GrayBytes();
        var destination = new byte[gray.Length];
        if (!NativeCore.TryDerivGauss(gray, destination, image.Width, image.Height, sigma, kind))
            DerivScalar(gray, destination, image.Width, image.Height, sigma, kind);
        return ImageBuffer.Wrap(image.Width, image.Height, 1, destination, null);
    }

    /// <summary>第一個字母沿列，第二個沿行。結果取絕對值，再除以兩個核的絕對值和與 2 的 Shift 次方。邊界用 REFLECT_101。</summary>
    /// <remarks>Laws 紋理能量。ll 大致保留原灰階，常數圖的 el 是 0。</remarks>
    /// <example>
    /// <code>Vizion.texture_laws (Image, ImageTexture, 'el', 0, 5)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="filterType">兩個字母。3 點可用 l、e、s。5 點再加 r、w。</param>
    /// <param name="shift">再除以 2 的這個次方，不能是負的。</param>
    /// <param name="filterSize">3 或 5。</param>
    /// <returns>紋理能量。不保留定義域。</returns>
    public static ImageBuffer TextureLaws(ImageBuffer image, string filterType, int shift, int filterSize)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (filterSize != 3 && filterSize != 5)
            throw new ArgumentException("FilterSize 只能是 3 或 5");
        if (shift < 0)
            throw new ArgumentException("Shift 不能是負的");
        int vertical = LawsCode(filterType, 0, filterSize);
        int horizontal = LawsCode(filterType, 1, filterSize);
        byte[] gray = image.GrayBytes();
        var destination = new byte[gray.Length];
        if (!NativeCore.TryTextureLaws(gray, destination, image.Width, image.Height, filterSize, vertical, horizontal, shift))
            LawsScalar(gray, destination, image.Width, image.Height, filterSize, vertical, horizontal, shift);
        return ImageBuffer.Wrap(image.Width, image.Height, 1, destination, null);
    }

    /// <summary>第 0 列是圓心。往下每一列半徑加 1。角度 0 朝右，正角度順時針，從左到右繞一圈。最近鄰，超出影像的地方補 0。</summary>
    /// <remarks>把圓周展開成矩形，方便沿角度量一圈的灰階。</remarks>
    /// <example>
    /// <code>Vizion.polar_trans_image (Image, PolarTransImage, 5, 5, 4, 4)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="row">圓心列。</param>
    /// <param name="column">圓心行。</param>
    /// <param name="width">一圈要切幾份，變成寬度。</param>
    /// <param name="height">半徑 0 到 Height-1，變成高度。</param>
    /// <returns>極座標影像。不保留定義域。</returns>
    public static ImageBuffer PolarTransImage(ImageBuffer image, double row, double column, int width, int height)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (width < 1 || height < 1)
            throw new ArgumentException("寬高至少要 1");
        var destination = new byte[width * height * image.Channels];
        if (!NativeCore.TryPolar(image.Pixels, destination, image.Width, image.Height, image.Channels, row, column, width, height))
            PolarScalar(image.Pixels, destination, image.Width, image.Height, image.Channels, row, column, width, height);
        return ImageBuffer.Wrap(width, height, image.Channels, destination, null);
    }

    /// <summary>後圖減前圖，再加 128。128 是沒變，比 128 大是變亮，比 128 小是變暗。彩色比較的是亮度。</summary>
    /// <remarks>看加強前後差在哪。</remarks>
    /// <example>
    /// <code>Vizion.enhance_map (Original, Enhanced, Map)</code>
    /// </example>
    /// <param name="original">原圖。</param>
    /// <param name="enhanced">加強後。</param>
    /// <returns>差圖。定義域沿用原圖。</returns>
    public static ImageBuffer EnhanceMap(ImageBuffer original, ImageBuffer enhanced)
    {
        byte[] before = PairGray(original, enhanced, out byte[] after);
        var destination = new byte[before.Length];
        if (!NativeCore.TryEnhanceMap(before, after, destination))
            MapScalar(before, after, destination);
        return ImageBuffer.Wrap(original.Width, original.Height, 1, destination, original.Domain);
    }

    /// <summary>兩個數都是絕對差。Mean 是 float32 累加後再除以像素數，Maximum 是最大的那一點。彩色比較的是亮度。</summary>
    /// <remarks>看加強一共動了多少。Mean 小、Maximum 集中在邊緣，表示平坦處沒有被一起放大。</remarks>
    /// <example>
    /// <code>Vizion.enhance_change (Original, Enhanced, Mean, Maximum)</code>
    /// </example>
    /// <param name="original">原圖。</param>
    /// <param name="enhanced">加強後。</param>
    /// <param name="mean">平均變了多少灰階。</param>
    /// <param name="maximum">最多變了多少灰階。</param>
    public static void EnhanceChange(ImageBuffer original, ImageBuffer enhanced, out double mean, out double maximum)
    {
        byte[] before = PairGray(original, enhanced, out byte[] after);
        if (!NativeCore.TryEnhanceChange(before, after, out mean, out maximum))
            ChangeScalar(before, after, out mean, out maximum);
    }

    /// <summary>變化的絕對值達到 Limit 才留下。0 表示只要有變到就算。彩色比較的是亮度。結果是單一區域。</summary>
    /// <remarks>標出被加強得最兇的地方。</remarks>
    /// <example>
    /// <code>Vizion.enhance_hot (Original, Enhanced, Hot, 10)</code>
    /// </example>
    /// <param name="original">原圖。</param>
    /// <param name="enhanced">加強後。</param>
    /// <param name="limit">至少變了多少灰階。</param>
    /// <returns>變動達到門檻的像素。</returns>
    public static Region EnhanceHot(ImageBuffer original, ImageBuffer enhanced, double limit)
    {
        if (limit < 0)
            throw new ArgumentException("Limit 不能是負的");
        byte[] before = PairGray(original, enhanced, out byte[] after);
        var mask = new byte[before.Length];
        if (!NativeCore.TryEnhanceHot(before, after, mask, limit))
            HotScalar(before, after, mask, limit);
        return Threshold(ImageBuffer.Gray(original.Width, original.Height, mask), 255, 255);
    }

    /// <summary>每個像素取絕對差。差不到 MinDiff 不算。定義域外不算。彩色先轉灰階。0 表示只要有差就算。</summary>
    /// <remarks>和黃金影像差在哪。</remarks>
    /// <example>
    /// <code>Vizion.diff_region (Image, Golden, Defects, 20)</code>
    /// </example>
    /// <param name="image">待測影像。</param>
    /// <param name="golden">黃金標準。</param>
    /// <param name="minimum">至少差幾個灰階才算。</param>
    /// <returns>差超過門檻的像素，合成一個區域。</returns>
    public static Region DiffRegion(ImageBuffer image, ImageBuffer golden, double minimum)
    {
        if (minimum < 0)
            throw new ArgumentException("MinDiff 不能是負的");
        byte[] left = PairGray(image, golden, out byte[] right);
        byte[]? domain = PairDomain(image, golden);
        var mask = new byte[left.Length];
        if (!NativeCore.TryDiffMask(left, right, domain, mask, minimum))
            DiffMaskScalar(left, right, domain, mask, minimum);
        return Threshold(ImageBuffer.Gray(image.Width, image.Height, mask), 255, 255);
    }

    /// <summary>Mean 是有效像素的平均絕對差，Maximum 是最大的那一點。定義域外不算。彩色先轉灰階。沒有有效像素時兩者都是 0。</summary>
    /// <remarks>和黃金影像差多少。</remarks>
    /// <example>
    /// <code>Vizion.diff_amount (Image, Golden, Mean, Maximum)</code>
    /// </example>
    /// <param name="image">待測影像。</param>
    /// <param name="golden">黃金標準。</param>
    /// <param name="mean">平均絕對差。</param>
    /// <param name="maximum">最大絕對差。</param>
    public static void DiffAmount(ImageBuffer image, ImageBuffer golden, out double mean, out double maximum)
    {
        byte[] left = PairGray(image, golden, out byte[] right);
        byte[]? domain = PairDomain(image, golden);
        if (!NativeCore.TryDiffAmount(left, right, domain, out mean, out maximum))
            DiffAmountScalar(left, right, domain, out mean, out maximum);
    }

    /// <summary>原圖減去高斯。絕對值小於 Threshold 的起伏當成 0。128 是沒有要加的地方。彩色用亮度。</summary>
    /// <remarks>看加強時加了什麼。這是 edge_enhance 在 Factor 為 1、Limit 為 0 時加回去的量。</remarks>
    /// <example>
    /// <code>Vizion.edge_residual (Image, Residual, 1.2, 2)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="sigma">邊緣大概多寬。0 是整張 128。</param>
    /// <param name="threshold">比這個小的起伏當成 0。</param>
    /// <returns>殘差影像。定義域沿用輸入。</returns>
    /// <summary>高斯模糊後，把原圖減模糊的差加回去。差的絕對值小於 Threshold 就不加。Limit 大於 0 時，加回去的量不會超過它。彩色只調亮度。</summary>
    /// <remarks>只加強夠明顯的邊緣。平坦處的小雜訊不會被放大。</remarks>
    /// <example>
    /// <code>Vizion.edge_enhance (Image, Enhanced, 1.2, 1, 2, 0)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="sigma">邊緣大概多寬。0 是不處理。</param>
    /// <param name="factor">加強多少。0 是原圖。</param>
    /// <param name="threshold">比這個小的起伏不加。</param>
    /// <param name="limit">單邊最多加或減多少。0 是不限制。</param>
    /// <returns>邊緣加強後的影像。定義域外的像素保持原值。</returns>
    public static ImageBuffer EdgeEnhance(ImageBuffer image, double sigma, double factor, double threshold, double limit)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (sigma < 0)
            throw new ArgumentException("Sigma 不能是負的");
        if (factor < 0)
            throw new ArgumentException("Factor 不能是負的");
        if (threshold < 0)
            throw new ArgumentException("Threshold 不能是負的");
        if (limit < 0)
            throw new ArgumentException("Limit 不能是負的");
        byte[]? domain = OwnDomain(image);
        var destination = new byte[image.Pixels.Length];
        if (!NativeCore.TryEdgeEnhance(image.Pixels, destination, image.Width, image.Height, image.Channels, sigma, factor, threshold, limit, domain))
            EnhanceScalar(image.Pixels, destination, image.Width, image.Height, image.Channels, sigma, factor, threshold, limit, domain);
        return ImageBuffer.Wrap(image.Width, image.Height, image.Channels, destination, image.Domain);
    }

    public static ImageBuffer EdgeResidual(ImageBuffer image, double sigma, double threshold)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (threshold < 0)
            throw new ArgumentException("Threshold 不能是負的");
        if (sigma < 0)
            throw new ArgumentException("Sigma 不能是負的");
        byte[] gray = image.GrayBytes();
        var destination = new byte[gray.Length];
        if (!NativeCore.TryEdgeResidual(gray, destination, image.Width, image.Height, sigma, threshold))
            ResidualScalar(gray, destination, image.Width, image.Height, sigma, threshold);
        return ImageBuffer.Wrap(image.Width, image.Height, 1, destination, image.Domain);
    }

    static int DerivCode(string component)
    {
        if (component == null)
            throw new ArgumentNullException(nameof(component));
        string name = component.Trim().Trim('\'').Trim('"').ToLowerInvariant();
        return name switch
        {
            "none" => 0,
            "x" => 1,
            "y" => 2,
            "xx" => 3,
            "yy" => 4,
            "xy" => 5,
            "gradient" => 6,
            "laplace" => 7,
            _ => throw new ArgumentException("Component 不支援"),
        };
    }

    static int LawsCode(string filterType, int place, int size)
    {
        if (filterType == null)
            throw new ArgumentNullException(nameof(filterType));
        string name = filterType.Trim().Trim('\'').Trim('"').ToLowerInvariant();
        if (name.Length != 2)
            throw new ArgumentException(LawsMessage(size));
        int code = LawsLetter(name[place]);
        if (code < 0 || (size == 3 && code > 2))
            throw new ArgumentException(LawsMessage(size));
        return code;
    }

    static int LawsLetter(char letter) => letter switch
    {
        'l' => 0,
        'e' => 1,
        's' => 2,
        'r' => 3,
        'w' => 4,
        _ => -1,
    };

    static string LawsMessage(int size) => size == 3
        ? "FilterType 要兩個字母，這個尺寸可用 e、l、s"
        : "FilterType 要兩個字母，這個尺寸可用 e、l、r、s、w";

    static byte[] PairGray(ImageBuffer left, ImageBuffer right, out byte[] other)
    {
        if (left == null)
            throw new ArgumentNullException(nameof(left));
        if (right == null)
            throw new ArgumentNullException(nameof(right));
        byte[] first = left.GrayBytes();
        other = right.GrayBytes();
        if (first.Length != other.Length || left.Width != right.Width || left.Height != right.Height)
            throw new ArgumentException("兩張影像的尺寸要相同");
        return first;
    }

    static byte[]? PairDomain(ImageBuffer left, ImageBuffer right)
    {
        if (left.Domain == null && right.Domain == null)
            return null;
        var mask = new byte[left.Width * left.Height];
        for (int i = 0; i < mask.Length; i++)
            mask[i] = 255;
        PaintDomain(mask, left);
        PaintDomain(mask, right);
        return mask;
    }

    static void PaintDomain(byte[] mask, ImageBuffer image)
    {
        if (image.Domain == null)
            return;
        var keep = new byte[mask.Length];
        for (int i = 0; i < image.Domain.Rows.Length; i++)
        {
            int row = image.Domain.Rows[i];
            int column = image.Domain.Columns[i];
            int length = image.Domain.Lengths[i];
            if ((uint)row >= (uint)image.Height || length <= 0)
                continue;
            int offset = row * image.Width + column;
            for (int step = 0; step < length && column + step < image.Width; step++)
                keep[offset + step] = 255;
        }
        for (int i = 0; i < mask.Length; i++)
            mask[i] = mask[i] != 0 && keep[i] != 0 ? (byte)255 : (byte)0;
    }

    static void DerivScalar(byte[] source, byte[] destination, int width, int height, double sigma, int component)
    {
        int count = KernelDeriv(sigma, out float[] smooth, out float[] first, out float[] second);
        var gray = new float[source.Length];
        for (int i = 0; i < source.Length; i++)
            gray[i] = source[i];
        float[] kx = smooth;
        float[] ky = smooth;
        if (component is 1 or 5 or 6)
            kx = first;
        else if (component is 3 or 7)
            kx = second;
        if (component is 2 or 5 or 6)
            ky = first;
        else if (component is 4 or 7)
            ky = second;
        if (component == 6)
        {
            kx = first;
            ky = smooth;
        }
        if (component == 7)
        {
            kx = second;
            ky = smooth;
        }
        float[] across = GradeSep(gray, width, height, kx, ky);
        float[]? other = null;
        if (component == 6)
            other = GradeSep(gray, width, height, smooth, first);
        else if (component == 7)
            other = GradeSep(gray, width, height, smooth, second);
        for (int i = 0; i < destination.Length; i++)
        {
            float response = across[i];
            if (component == 6)
                response = (float)Math.Sqrt(across[i] * across[i] + other![i] * other[i]);
            else if (component == 7)
                response += other![i];
            if (component != 0 && component != 6)
                response += 128f;
            destination[i] = RoundF(response);
        }
        _ = count;
    }

    static int KernelDeriv(double sigma, out float[] smooth, out float[] first, out float[] second)
    {
        int radius = (int)Math.Ceiling(sigma * 4.0);
        if (radius < 1)
            radius = 1;
        int count = radius * 2 + 1;
        smooth = new float[count];
        first = new float[count];
        second = new float[count];
        double variance = sigma * sigma;
        double sum = 0;
        var raw = new double[count];
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
            first[at] = (float)(i / variance * weight);
            second[at] = (float)((i * (double)i / (variance * variance) - 1.0 / variance) * weight);
        }
        return count;
    }

    static readonly float[][] Laws3 =
    [
        [1f, 2f, 1f],
        [-1f, 0f, 1f],
        [-1f, 2f, -1f],
    ];
    static readonly float[][] Laws5 =
    [
        [1f, 4f, 6f, 4f, 1f],
        [-1f, -2f, 0f, 2f, 1f],
        [-1f, 0f, 2f, 0f, -1f],
        [1f, -4f, 6f, -4f, 1f],
        [-1f, 2f, 0f, -2f, 1f],
    ];

    static void LawsScalar(byte[] source, byte[] destination, int width, int height, int size, int vertical, int horizontal, int shift)
    {
        float[] down = (size == 3 ? Laws3 : Laws5)[vertical];
        float[] side = (size == 3 ? Laws3 : Laws5)[horizontal];
        float vsum = 0f;
        float hsum = 0f;
        for (int i = 0; i < down.Length; i++)
        {
            vsum += Math.Abs(down[i]);
            hsum += Math.Abs(side[i]);
        }
        double scale = (double)(vsum * hsum) * Math.Pow(2.0, shift);
        var gray = new float[source.Length];
        for (int i = 0; i < source.Length; i++)
            gray[i] = source[i];
        float[] response = GradeSep(gray, width, height, side, down);
        for (int i = 0; i < destination.Length; i++)
            destination[i] = RoundD(Math.Abs(response[i]) / scale);
    }

    static void PolarScalar(byte[] source, byte[] destination, int width, int height, int channels, double row, double column, int outWidth, int outHeight)
    {
        double turn = 2.0 * Math.Acos(-1.0) / outWidth;
        for (int y = 0; y < outHeight; y++)
        {
            float radius = y;
            for (int x = 0; x < outWidth; x++)
            {
                float angle = (float)(x * turn);
                float cosine = (float)Math.Cos(angle);
                float sine = (float)Math.Sin(angle);
                float mapX = (float)(column + radius * cosine);
                float mapY = (float)(row + radius * sine);
                int sampleX = Nearest(mapX);
                int sampleY = Nearest(mapY);
                bool inside = (uint)sampleX < (uint)width && (uint)sampleY < (uint)height;
                for (int channel = 0; channel < channels; channel++)
                {
                    byte value = 0;
                    if (inside)
                        value = source[(sampleY * width + sampleX) * channels + channel];
                    destination[(y * outWidth + x) * channels + channel] = value;
                }
            }
        }
    }

    static int Nearest(float value) => (int)Math.Round((double)value, MidpointRounding.ToEven);

    static void MapScalar(byte[] before, byte[] after, byte[] destination)
    {
        for (int i = 0; i < destination.Length; i++)
            destination[i] = RoundF(128f + (after[i] - before[i]));
    }

    static void ChangeScalar(byte[] before, byte[] after, out double mean, out double maximum)
    {
        float sum = 0f;
        float peak = 0f;
        for (int i = 0; i < before.Length; i++)
        {
            float delta = after[i] - before[i];
            if (delta < 0f)
                delta = -delta;
            sum += delta;
            if (delta > peak)
                peak = delta;
        }
        mean = sum / before.Length;
        maximum = peak;
    }

    static void HotScalar(byte[] before, byte[] after, byte[] mask, double limit)
    {
        for (int i = 0; i < mask.Length; i++)
        {
            float delta = after[i] - before[i];
            if (delta < 0f)
                delta = -delta;
            mask[i] = limit == 0.0 ? (delta > 0f ? (byte)255 : (byte)0) : ((double)delta + 1e-6 >= limit ? (byte)255 : (byte)0);
        }
    }

    static void DiffMaskScalar(byte[] image, byte[] golden, byte[]? domain, byte[] mask, double minimum)
    {
        for (int i = 0; i < mask.Length; i++)
        {
            if (domain != null && domain[i] == 0)
            {
                mask[i] = 0;
                continue;
            }
            int delta = image[i] - golden[i];
            if (delta < 0)
                delta = -delta;
            mask[i] = minimum == 0.0 ? (delta > 0 ? (byte)255 : (byte)0) : ((double)delta + 1e-6 >= minimum ? (byte)255 : (byte)0);
        }
    }

    static void DiffAmountScalar(byte[] image, byte[] golden, byte[]? domain, out double mean, out double maximum)
    {
        double sum = 0;
        double peak = 0;
        int used = 0;
        for (int i = 0; i < image.Length; i++)
        {
            if (domain != null && domain[i] == 0)
                continue;
            int delta = image[i] - golden[i];
            if (delta < 0)
                delta = -delta;
            sum += delta;
            if (delta > peak)
                peak = delta;
            used++;
        }
        mean = used == 0 ? 0 : sum / used;
        maximum = peak;
    }

    static byte[]? OwnDomain(ImageBuffer image)
    {
        if (image.Domain == null)
            return null;
        var mask = new byte[image.Width * image.Height];
        for (int i = 0; i < image.Domain.Rows.Length; i++)
        {
            int row = image.Domain.Rows[i];
            int column = image.Domain.Columns[i];
            int length = image.Domain.Lengths[i];
            if ((uint)row >= (uint)image.Height || length <= 0)
                continue;
            int offset = row * image.Width + column;
            for (int step = 0; step < length && column + step < image.Width; step++)
                mask[offset + step] = 255;
        }
        return mask;
    }

    static void EnhanceScalar(
        byte[] source, byte[] destination, int width, int height, int channels,
        double sigma, double factor, double threshold, double limit, byte[]? domain)
    {
        int pixels = width * height;
        if (!(sigma > 0) || !(factor > 0))
        {
            Buffer.BlockCopy(source, 0, destination, 0, source.Length);
            return;
        }
        var luma = new float[pixels];
        if (channels == 1)
        {
            for (int i = 0; i < pixels; i++)
                luma[i] = source[i];
        }
        else
        {
            for (int i = 0; i < pixels; i++)
            {
                BgrToYCrCb(source[i * 3], source[i * 3 + 1], source[i * 3 + 2], out int y, out _, out _);
                luma[i] = y;
            }
        }
        int side = BlurSide(sigma);
        float[] kernel = GaussKernel(sigma, side);
        float[] blurred = GradeSep(luma, width, height, kernel, kernel);
        if (channels == 1)
        {
            for (int i = 0; i < pixels; i++)
                destination[i] = RoundF(luma[i] + Boost(luma[i], blurred[i], factor, threshold, limit));
        }
        else
        {
            Buffer.BlockCopy(source, 0, destination, 0, source.Length);
            for (int i = 0; i < pixels; i++)
            {
                float boost = Boost(luma[i], blurred[i], factor, threshold, limit);
                if (!(boost > 1e-6f || boost < -1e-6f))
                    continue;
                BgrToYCrCb(source[i * 3], source[i * 3 + 1], source[i * 3 + 2], out int y, out int cr, out int cb);
                YCrCbToBgr((int)RoundF(luma[i] + boost), cr, cb, destination, i * 3);
            }
        }
        if (domain != null)
        {
            for (int i = 0; i < pixels; i++)
            {
                if (domain[i] != 0)
                    continue;
                for (int channel = 0; channel < channels; channel++)
                    destination[i * channels + channel] = source[i * channels + channel];
            }
        }
    }

    static float Boost(float sample, float blurred, double factor, double threshold, double limit)
    {
        float boost = sample - blurred;
        if (threshold > 0 && !(Math.Abs(boost) >= threshold))
            boost = 0f;
        boost *= (float)factor;
        if (limit > 0)
        {
            float cap = (float)limit;
            if (boost > cap)
                boost = cap;
            if (boost < -cap)
                boost = -cap;
        }
        return boost;
    }

    static void ResidualScalar(byte[] source, byte[] destination, int width, int height, double sigma, double threshold)
    {
        if (!(sigma > 0))
        {
            for (int i = 0; i < destination.Length; i++)
                destination[i] = 128;
            return;
        }
        int side = BlurSide(sigma);
        float[] kernel = GaussKernel(sigma, side);
        var gray = new float[source.Length];
        for (int i = 0; i < source.Length; i++)
            gray[i] = source[i];
        float[] blurred = GradeSep(gray, width, height, kernel, kernel);
        for (int i = 0; i < destination.Length; i++)
        {
            float boost = gray[i] - blurred[i];
            if (threshold > 0 && !(Math.Abs(boost) >= threshold))
                boost = 0f;
            destination[i] = RoundF(128f + boost);
        }
    }

    static int BlurSide(double sigma)
    {
        int side = GradeBankers(sigma * 6.0) | 1;
        if (side < 3)
            side = 3;
        if (side > 31)
            side = 31;
        return side;
    }

    static int GradeBankers(double value)
    {
        int whole = (int)Math.Floor(value);
        double frac = value - whole;
        if (frac > 0.5)
            return whole + 1;
        if (frac < 0.5)
            return whole;
        return (whole & 1) != 0 ? whole + 1 : whole;
    }

    static float[] GaussKernel(double sigma, int ksize)
    {
        double scale = -0.5 / (sigma * sigma);
        double half = (ksize - 1) * 0.5;
        var raw = new double[ksize];
        double sum = 0;
        for (int i = 0; i < ksize; i++)
        {
            double x = i - half;
            raw[i] = Math.Exp(scale * x * x);
            sum += raw[i];
        }
        var kernel = new float[ksize];
        for (int i = 0; i < ksize; i++)
            kernel[i] = (float)(raw[i] / sum);
        return kernel;
    }

    static float[] GradeSep(float[] source, int width, int height, float[] across, float[] along)
    {
        var horizontal = new float[source.Length];
        var destination = new float[source.Length];
        var column = new float[height];
        var filtered = new float[height];
        for (int row = 0; row < height; row++)
            Conv(source, row * width, 1, width, horizontal, row * width, across);
        for (int col = 0; col < width; col++)
        {
            for (int row = 0; row < height; row++)
                column[row] = horizontal[row * width + col];
            Conv(column, 0, 1, height, filtered, 0, along);
            for (int row = 0; row < height; row++)
                destination[row * width + col] = filtered[row];
        }
        return destination;
    }

    static void Conv(float[] source, int sourceAt, int stride, int length, float[] destination, int destAt, float[] kernel)
    {
        int anchor = kernel.Length / 2;
        for (int index = 0; index < length; index++)
        {
            float acc = 0f;
            for (int tap = 0; tap < kernel.Length; tap++)
            {
                int sample = GradeReflect(index - anchor + tap, length);
                acc += source[sourceAt + sample * stride] * kernel[tap];
            }
            destination[destAt + index] = acc;
        }
    }

    static int GradeReflect(int position, int length)
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
}
