namespace Vizion;

public static partial class Operators
{
    /// <summary>light：原圖大於等於平滑圖加 Offset。dark：原圖小於等於平滑圖減 Offset。equal：絕對差不超過 Offset。not_equal 是 equal 的相反。</summary>
    /// <remarks>和另一張圖比較的局部閾值。</remarks>
    /// <example>
    /// <code>Vizion.dyn_threshold (Image, Smooth, Region, 5, 'light')</code>
    /// </example>
    /// <param name="image">原圖。</param>
    /// <param name="smooth">比較用的平滑圖。</param>
    /// <param name="offset">灰階差。</param>
    /// <param name="lightDark">light、dark、equal 或 not_equal。</param>
    /// <returns>局部閾值區域。只留原圖定義域裡的像素。</returns>
    public static Region DynThreshold(ImageBuffer image, ImageBuffer smooth, double offset, string lightDark)
    {
        if (image == null || smooth == null)
            throw new ArgumentNullException(nameof(image));
        if (image.Width != smooth.Width || image.Height != smooth.Height)
            throw new ArgumentException("兩張影像的尺寸不同");
        int mode = LightCode(lightDark);
        byte[] left = image.GrayBytes();
        byte[] right = smooth.GrayBytes();
        byte[]? domain = OwnDomain(image);
        var mask = new byte[left.Length];
        if (!NativeCore.TryDynThreshold(left, right, domain, mask, offset, mode))
            DynScalar(left, right, domain, mask, offset, mode);
        return RunsOf(image.Width, image.Height, mask);
    }

    /// <summary>dark 留下比局部平均還低超過 Offset 的像素。light 留下比局部平均還高超過 Offset 的像素。彩色先轉成亮度。</summary>
    /// <remarks>用局部平均做門檻。</remarks>
    /// <example>
    /// <code>Vizion.local_threshold (Image, Region, 'mean', 'dark', 3, 20)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="method">這版做 mean。</param>
    /// <param name="lightDark">dark 或 light。</param>
    /// <param name="maskSize">局部視窗，奇數。</param>
    /// <param name="offset">至少要偏離平均多少。</param>
    /// <returns>偏離局部平均的像素。</returns>
    public static Region LocalThreshold(ImageBuffer image, string method, string lightDark, int maskSize, double offset)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (MeanOnly(method) != 0)
            throw new ArgumentException("Method 這版只做 mean");
        int light = DarkOrLight(lightDark);
        if (maskSize < 1 || (maskSize & 1) == 0)
            throw new ArgumentException("MaskSize 要是正的奇數");
        byte[] gray = image.GrayBytes();
        var mask = new byte[gray.Length];
        if (!NativeCore.TryLocalThreshold(gray, mask, image.Width, image.Height, maskSize, offset, light))
            LocalMeanScalar(gray, mask, image.Width, image.Height, maskSize, offset, light != 0);
        return RunsOf(image.Width, image.Height, mask);
    }

    /// <summary>鄰域裡每個像素都一樣才算。大片相同背景的內部也會被標出來。偶數會加 1，最大 15。彩色先轉成亮度。</summary>
    /// <remarks>找出平坦的灰階塊。</remarks>
    /// <example>
    /// <code>Vizion.plateaus (Image, Plateaus, 3)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="size">鄰域邊長，奇數。</param>
    /// <returns>平坦區塊。有定義域時只留定義域裡的像素。</returns>
    public static Region Plateaus(ImageBuffer image, int size)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        int side = size | 1;
        if (side < 3 || side > 15)
            throw new ArgumentException("Size 要在 3 到 15");
        byte[] gray = image.GrayBytes();
        byte[]? domain = OwnDomain(image);
        var mask = new byte[gray.Length];
        if (!NativeCore.TryPlateau(gray, domain, mask, image.Width, image.Height, side))
            PlateauScalar(gray, domain, mask, image.Width, image.Height, side);
        return RunsOf(image.Width, image.Height, mask);
    }

    /// <summary>binary 留下大於 Threshold 的像素，binary_inv 留下小於等於 Threshold 的像素。等於門檻的像素在 binary 不會留下。</summary>
    /// <remarks>OpenCV 二值化。</remarks>
    /// <example>
    /// <code>Vizion.opencv_threshold (Image, Region, 128, 'binary')</code>
    /// </example>
    /// <param name="image">灰階或彩色影像。</param>
    /// <param name="threshold">門檻，不含等於。</param>
    /// <param name="method">binary 或 binary_inv。</param>
    /// <returns>二值化區域。有定義域時只留定義域裡的像素。</returns>
    public static Region OpenCvThreshold(ImageBuffer image, int threshold, string method)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (threshold < 0 || threshold > 255)
            throw new ArgumentException("Threshold 要在 0 到 255");
        int inverse = BinaryCode(method);
        byte[] gray = image.GrayBytes();
        byte[]? domain = OwnDomain(image);
        var mask = new byte[gray.Length];
        if (!NativeCore.TryOpenCvThreshold(gray, domain, mask, threshold, inverse))
            OpenCvScalar(gray, domain, mask, threshold, inverse != 0);
        return RunsOf(image.Width, image.Height, mask);
    }

    /// <summary>整張圖找一個門檻，留下比門檻亮的像素。全圖同一個灰階時區域是空的。彩色先轉成亮度。</summary>
    /// <remarks>用 Otsu 自動分成黑白。</remarks>
    /// <example>
    /// <code>Vizion.bin_threshold (Image, Region)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <returns>較亮的那一邊。</returns>
    public static Region BinThreshold(ImageBuffer image)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        byte[] gray = image.GrayBytes();
        var mask = new byte[gray.Length];
        if (!NativeCore.TryBinThreshold(gray, mask))
            BinScalar(gray, mask);
        return RunsOf(image.Width, image.Height, mask);
    }

    /// <summary>高門檻以上是種子，低門檻以上而且和種子八連通的像素留下。MaxLength 大於 0 時，種子先膨脹這個半徑再去接低門檻。</summary>
    /// <remarks>滯後閾值。</remarks>
    /// <example>
    /// <code>Vizion.hysteresis_threshold (Image, Region, 20, 40, 0)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="low">低門檻。</param>
    /// <param name="high">高門檻，種子。</param>
    /// <param name="maxLength">種子可以跨出去接低門檻的半徑。</param>
    /// <returns>滯後閾值區域。只留定義域裡的像素。</returns>
    public static Region HysteresisThreshold(ImageBuffer image, double low, double high, double maxLength)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (high < low)
            throw new ArgumentException("High 不能小於 Low");
        if (maxLength < 0)
            throw new ArgumentException("MaxLength 不能是負的");
        byte[] gray = image.GrayBytes();
        byte[]? domain = OwnDomain(image);
        var mask = new byte[gray.Length];
        if (!NativeCore.TryHysteresis(gray, domain, mask, image.Width, image.Height, low, high, maxLength))
            HysteresisScalar(gray, domain, mask, image.Width, image.Height, low, high, maxLength);
        return RunsOf(image.Width, image.Height, mask);
    }

    /// <summary>門檻是 max(標準差 × StdDevScale, AbsThreshold)。light 取高於平均加門檻的像素，dark 取低於平均減門檻的像素。</summary>
    /// <remarks>用局部平均和標準差做閾值。</remarks>
    /// <example>
    /// <code>Vizion.var_threshold (Image, Region, 15, 15, 0.2, 2, 'light')</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="maskWidth">局部視窗寬，奇數。</param>
    /// <param name="maskHeight">局部視窗高，奇數。</param>
    /// <param name="stdDevScale">標準差倍率。</param>
    /// <param name="absThreshold">至少要差這麼多灰階。</param>
    /// <param name="lightDark">light、dark、equal 或 not_equal。</param>
    /// <returns>局部閾值區域。只留定義域裡的像素。</returns>
    public static Region VarThreshold(
        ImageBuffer image, int maskWidth, int maskHeight, double stdDevScale, double absThreshold, string lightDark)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (maskWidth < 1 || maskHeight < 1)
            throw new ArgumentException("視窗寬高至少要 1");
        if ((maskWidth & 1) == 0)
            maskWidth++;
        if ((maskHeight & 1) == 0)
            maskHeight++;
        int mode = LightCode(lightDark);
        byte[] gray = image.GrayBytes();
        byte[]? domain = OwnDomain(image);
        var mask = new byte[gray.Length];
        if (!NativeCore.TryVarThreshold(gray, domain, mask, image.Width, image.Height, maskWidth, maskHeight, stdDevScale, absThreshold, mode))
            VarScalar(gray, domain, mask, image.Width, image.Height, maskWidth, maskHeight, stdDevScale, absThreshold, mode);
        return RunsOf(image.Width, image.Height, mask);
    }

    /// <summary>門檻是局部平均乘上 1 加 K 乘（局部標準差除以 128 再減 1）。比這個門檻暗的像素留下。彩色先轉成亮度。</summary>
    /// <remarks>Sauvola 局部門檻。</remarks>
    /// <example>
    /// <code>Vizion.sauvola (Image, Region, 15, 0.5)</code>
    /// </example>
    /// <param name="image">灰階或彩色影像。</param>
    /// <param name="maskSize">局部視窗，奇數。</param>
    /// <param name="k">對比係數。</param>
    /// <returns>比局部門檻暗的像素。有定義域時只留定義域裡的像素。</returns>
    public static Region Sauvola(ImageBuffer image, int maskSize, double k)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (maskSize < 3 || (maskSize & 1) == 0)
            throw new ArgumentException("MaskSize 要是 3 以上的奇數");
        if (!(k > 0) || k > 1)
            throw new ArgumentException("K 要大於 0，而且不大於 1");
        byte[] gray = image.GrayBytes();
        byte[]? domain = OwnDomain(image);
        var mask = new byte[gray.Length];
        if (!NativeCore.TrySauvola(gray, domain, mask, image.Width, image.Height, maskSize, k))
            SauvolaScalar(gray, domain, mask, image.Width, image.Height, maskSize, k);
        return RunsOf(image.Width, image.Height, mask);
    }

    /// <summary>高門檻以上是種子的來源。比 Threshold 更亮或更暗的像素做八連通，面積夠而且裡面有更強的像素才留下。彩色先轉成亮度。</summary>
    /// <remarks>雙閾值。灰階 128 當作 0。</remarks>
    /// <example>
    /// <code>Vizion.dual_threshold (Image, RegionCrossings, 4, 40, 20)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="minSize">最小面積。</param>
    /// <param name="minGray">區塊裡至少要有一個這麼強的像素。</param>
    /// <param name="threshold">候選像素的最低強度。</param>
    /// <returns>留下的區塊。每一塊一個區域。</returns>
    public static Region[] DualThreshold(ImageBuffer image, int minSize, double minGray, double threshold)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (threshold < 0 || minGray < 0)
            throw new ArgumentException("Threshold 和 MinGray 不能是負的");
        if (minSize < 0)
            throw new ArgumentException("MinSize 不能是負的");
        byte[] gray = image.GrayBytes();
        byte[]? domain = OwnDomain(image);
        var labels = new int[gray.Length];
        int count;
        if (!NativeCore.TryDualLabels(gray, domain, labels, image.Width, image.Height, threshold, minGray, minSize, out count))
            count = DualScalar(gray, domain, labels, image.Width, image.Height, threshold, minGray, minSize);
        return SplitLabels(image.Width, image.Height, labels, count);
    }

    /// <summary>只從給的種子開始。八鄰域裡，和已經長到的那個像素灰階差不超過 Tolerance 就加入。每個種子一塊區域。</summary>
    /// <remarks>從種子做區域成長。</remarks>
    /// <example>
    /// <code>Vizion.regiongrowing (Image, Regions, 120, 100, 10, 20)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="rows">種子列。</param>
    /// <param name="columns">種子行。</param>
    /// <param name="tolerance">相鄰像素允許的灰階差。</param>
    /// <param name="minSize">小於這個面積就丟掉。</param>
    /// <returns>長出來的區域。每一塊一個區域。</returns>
    public static Region[] RegionGrowing(ImageBuffer image, double[] rows, double[] columns, double tolerance, int minSize)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (rows == null || columns == null)
            throw new ArgumentNullException(nameof(rows));
        if (rows.Length != columns.Length)
            throw new ArgumentException("Row 和 Column 的長度要相同");
        if (rows.Length == 0)
            throw new ArgumentException("至少要一個種子");
        var seedRows = new int[rows.Length];
        var seedColumns = new int[columns.Length];
        for (int i = 0; i < rows.Length; i++)
        {
            seedRows[i] = (int)Math.Round(rows[i]);
            seedColumns[i] = (int)Math.Round(columns[i]);
        }
        byte[] gray = image.GrayBytes();
        byte[]? domain = OwnDomain(image);
        var labels = new int[gray.Length];
        int count;
        if (!NativeCore.TryRegionGrow(gray, domain, labels, image.Width, image.Height, seedRows, seedColumns, tolerance, minSize, out count))
            count = GrowScalar(gray, domain, labels, image.Width, image.Height, seedRows, seedColumns, tolerance, minSize);
        return SplitLabels(image.Width, image.Height, labels, count);
    }

    static int LightCode(string mode)
    {
        if (mode == null)
            throw new ArgumentNullException(nameof(mode));
        string name = mode.Trim().Trim('\'').Trim('"').ToLowerInvariant();
        if (name == "light")
            return 0;
        if (name == "dark")
            return 1;
        if (name == "equal")
            return 2;
        if (name == "not_equal")
            return 3;
        throw new ArgumentException("LightDark 只能是 light、dark、equal 或 not_equal");
    }

    static int DarkOrLight(string mode)
    {
        if (mode == null)
            throw new ArgumentNullException(nameof(mode));
        string name = mode.Trim().Trim('\'').Trim('"').ToLowerInvariant();
        if (name == "dark")
            return 0;
        if (name == "light")
            return 1;
        throw new ArgumentException("LightDark 只能是 dark 或 light");
    }

    static int MeanOnly(string method)
    {
        if (method == null)
            throw new ArgumentNullException(nameof(method));
        string name = method.Trim().Trim('\'').Trim('"').ToLowerInvariant();
        if (name == "mean")
            return 0;
        throw new ArgumentException("Method 這版只做 mean");
    }

    static int BinaryCode(string method)
    {
        if (method == null)
            throw new ArgumentNullException(nameof(method));
        string name = method.Trim().Trim('\'').Trim('"').ToLowerInvariant();
        if (name == "binary")
            return 0;
        if (name == "binary_inv")
            return 1;
        throw new ArgumentException("Method 只能是 binary 或 binary_inv");
    }

    static Region[] SplitLabels(int width, int height, int[] labels, int count)
    {
        var regions = new Region[count];
        for (int label = 1; label <= count; label++)
        {
            var mask = new byte[labels.Length];
            for (int i = 0; i < labels.Length; i++)
            {
                if (labels[i] == label)
                    mask[i] = 255;
            }
            regions[label - 1] = RunsOf(width, height, mask);
        }
        return regions;
    }

    static void DynScalar(byte[] image, byte[] smooth, byte[]? domain, byte[] mask, double offset, int mode)
    {
        for (int i = 0; i < image.Length; i++)
        {
            if (domain != null && domain[i] == 0)
            {
                mask[i] = 0;
                continue;
            }
            int delta = image[i] - smooth[i];
            int absDelta = Math.Abs(delta);
            bool keep = mode == 0 ? image[i] >= smooth[i] + offset
                : mode == 1 ? image[i] <= smooth[i] - offset
                : mode == 2 ? absDelta <= offset
                : absDelta > offset;
            mask[i] = keep ? (byte)255 : (byte)0;
        }
    }

    static void BoxRect(float[] source, float[] destination, int width, int height, int maskWidth, int maskHeight)
    {
        int radiusX = maskWidth / 2;
        int radiusY = maskHeight / 2;
        var horizontal = new double[width * height];
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                double acc = 0;
                for (int tap = -radiusX; tap <= radiusX; tap++)
                    acc += source[row * width + GradeReflect(col + tap, width)];
                horizontal[row * width + col] = acc / maskWidth;
            }
        }
        for (int col = 0; col < width; col++)
        {
            for (int row = 0; row < height; row++)
            {
                double acc = 0;
                for (int tap = -radiusY; tap <= radiusY; tap++)
                    acc += horizontal[GradeReflect(row + tap, height) * width + col];
                destination[row * width + col] = (float)(acc / maskHeight);
            }
        }
    }

    static void LocalMeanScalar(byte[] gray, byte[] mask, int width, int height, int side, double offset, bool light)
    {
        var values = new float[gray.Length];
        var mean = new float[gray.Length];
        for (int i = 0; i < gray.Length; i++)
            values[i] = gray[i];
        BoxRect(values, mean, width, height, side, side);
        for (int i = 0; i < gray.Length; i++)
        {
            bool keep = light
                ? values[i] > mean[i] + offset + 1e-6
                : values[i] + 1e-6 < mean[i] - offset;
            mask[i] = keep ? (byte)255 : (byte)0;
        }
    }

    static void PlateauScalar(byte[] gray, byte[]? domain, byte[] mask, int width, int height, int side)
    {
        int radius = side / 2;
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                int value = gray[row * width + column];
                int highest = 0;
                int lowest = 255;
                for (int dy = -radius; dy <= radius; dy++)
                {
                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        int rr = row + dy;
                        int cc = column + dx;
                        bool outside = rr < 0 || cc < 0 || rr >= height || cc >= width;
                        int highSample = outside ? 0 : gray[rr * width + cc];
                        int lowSample = outside ? 255 : gray[rr * width + cc];
                        if (highSample > highest)
                            highest = highSample;
                        if (lowSample < lowest)
                            lowest = lowSample;
                    }
                }
                bool inside = domain == null || domain[row * width + column] != 0;
                mask[row * width + column] = value == highest && value == lowest && inside ? (byte)255 : (byte)0;
            }
        }
    }

    static void OpenCvScalar(byte[] gray, byte[]? domain, byte[] mask, int level, bool inverse)
    {
        for (int i = 0; i < gray.Length; i++)
        {
            bool keep = inverse ? gray[i] <= level : gray[i] > level;
            if (domain != null && domain[i] == 0)
                keep = false;
            mask[i] = keep ? (byte)255 : (byte)0;
        }
    }

    static int Otsu(byte[] gray)
    {
        var hist = new double[256];
        for (int i = 0; i < gray.Length; i++)
            hist[gray[i]]++;
        double sumAll = 0;
        for (int i = 0; i < 256; i++)
            sumAll += i * hist[i];
        double sumB = 0;
        double weightB = 0;
        double best = -1;
        int threshold = 0;
        for (int i = 0; i < 256; i++)
        {
            weightB += hist[i];
            if (weightB == 0)
                continue;
            double weightF = gray.Length - weightB;
            if (weightF == 0)
                break;
            sumB += i * hist[i];
            double meanB = sumB / weightB;
            double meanF = (sumAll - sumB) / weightF;
            double between = weightB * weightF * (meanB - meanF) * (meanB - meanF);
            if (between > best)
            {
                best = between;
                threshold = i;
            }
        }
        return threshold;
    }

    static void BinScalar(byte[] gray, byte[] mask)
    {
        int low = gray[0];
        int high = gray[0];
        for (int i = 1; i < gray.Length; i++)
        {
            if (gray[i] < low)
                low = gray[i];
            if (gray[i] > high)
                high = gray[i];
        }
        if (low == high)
        {
            Array.Clear(mask, 0, mask.Length);
            return;
        }
        int level = Otsu(gray);
        for (int i = 0; i < gray.Length; i++)
            mask[i] = gray[i] > level ? (byte)255 : (byte)0;
    }

    static void HysteresisScalar(byte[] gray, byte[]? domain, byte[] mask, int width, int height, double low, double high, double maxLength)
    {
        var lowMask = new byte[gray.Length];
        var seeds = new byte[gray.Length];
        for (int i = 0; i < gray.Length; i++)
        {
            if (domain != null && domain[i] == 0)
                continue;
            if (gray[i] >= low)
                lowMask[i] = 255;
            if (gray[i] >= high)
                seeds[i] = 255;
        }
        if (maxLength > 0)
        {
            int radius = (int)Math.Round(maxLength);
            if (radius < 1)
                radius = 1;
            seeds = DilateEllipse(seeds, width, height, radius);
        }
        Array.Clear(mask, 0, mask.Length);
        for (int i = 0; i < gray.Length; i++)
        {
            if (seeds[i] != 0 && lowMask[i] != 0 && mask[i] == 0)
                Flood(lowMask, mask, width, height, i);
        }
    }

    static byte[] EllipseKernel(int side)
    {
        var kernel = new byte[side * side];
        int radius = side / 2;
        double inverse = radius == 0 ? 0 : 1.0 / ((double)radius * radius);
        for (int row = 0; row < side; row++)
        {
            int dy = row - radius;
            double inside = ((double)radius * radius - (double)dy * dy) * inverse;
            if (inside < 0)
                inside = 0;
            int reach = (int)((double)radius * Math.Sqrt(inside) + 0.5);
            int begin = radius - reach;
            int end = radius + reach + 1;
            if (begin < 0)
                begin = 0;
            if (end > side)
                end = side;
            for (int col = begin; col < end; col++)
                kernel[row * side + col] = 1;
        }
        return kernel;
    }

    static byte[] DilateEllipse(byte[] source, int width, int height, int radius)
    {
        int side = radius * 2 + 1;
        var kernel = EllipseKernel(side);
        var destination = new byte[source.Length];
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                bool hot = false;
                for (int dy = -radius; dy <= radius && !hot; dy++)
                {
                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        if (kernel[(dy + radius) * side + dx + radius] == 0)
                            continue;
                        int rr = row + dy;
                        int cc = column + dx;
                        if (rr < 0 || cc < 0 || rr >= height || cc >= width)
                            continue;
                        if (source[rr * width + cc] != 0)
                        {
                            hot = true;
                            break;
                        }
                    }
                }
                destination[row * width + column] = hot ? (byte)255 : (byte)0;
            }
        }
        return destination;
    }

    static void Flood(byte[] allow, byte[] mask, int width, int height, int start)
    {
        var queue = new Queue<int>();
        int[] drow = { -1, -1, -1, 0, 0, 1, 1, 1 };
        int[] dcol = { -1, 0, 1, -1, 1, -1, 0, 1 };
        mask[start] = 255;
        queue.Enqueue(start);
        while (queue.Count > 0)
        {
            int index = queue.Dequeue();
            int row = index / width;
            int column = index - row * width;
            for (int neighbor = 0; neighbor < 8; neighbor++)
            {
                int nr = row + drow[neighbor];
                int nc = column + dcol[neighbor];
                if (nr < 0 || nc < 0 || nr >= height || nc >= width)
                    continue;
                int next = nr * width + nc;
                if (allow[next] == 0 || mask[next] != 0)
                    continue;
                mask[next] = 255;
                queue.Enqueue(next);
            }
        }
    }

    static void Stats(float[] values, float[] mean, float[] deviation, int width, int height, int maskWidth, int maskHeight)
    {
        var square = new float[values.Length];
        var meanSq = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
            square[i] = values[i] * values[i];
        BoxRect(values, mean, width, height, maskWidth, maskHeight);
        BoxRect(square, meanSq, width, height, maskWidth, maskHeight);
        for (int i = 0; i < values.Length; i++)
        {
            float var32 = meanSq[i] - mean[i] * mean[i];
            double spread = var32;
            if (spread < 0)
                spread = 0;
            deviation[i] = (float)Math.Sqrt(spread);
        }
    }

    static void VarScalar(
        byte[] gray, byte[]? domain, byte[] mask, int width, int height,
        int maskWidth, int maskHeight, double scale, double absolute, int mode)
    {
        var values = new float[gray.Length];
        var mean = new float[gray.Length];
        var deviation = new float[gray.Length];
        for (int i = 0; i < gray.Length; i++)
            values[i] = gray[i];
        Stats(values, mean, deviation, width, height, maskWidth, maskHeight);
        for (int i = 0; i < gray.Length; i++)
        {
            if (domain != null && domain[i] == 0)
            {
                mask[i] = 0;
                continue;
            }
            double limit = deviation[i] * scale;
            if (absolute > limit)
                limit = absolute;
            bool keep = mode == 0 ? values[i] >= mean[i] + limit
                : mode == 1 ? values[i] <= mean[i] - limit
                : mode == 2 ? Math.Abs(values[i] - mean[i]) <= limit
                : Math.Abs(values[i] - mean[i]) > limit;
            mask[i] = keep ? (byte)255 : (byte)0;
        }
    }

    static void SauvolaScalar(byte[] gray, byte[]? domain, byte[] mask, int width, int height, int side, double factor)
    {
        var values = new float[gray.Length];
        var mean = new float[gray.Length];
        var deviation = new float[gray.Length];
        for (int i = 0; i < gray.Length; i++)
            values[i] = gray[i];
        Stats(values, mean, deviation, width, height, side, side);
        for (int i = 0; i < gray.Length; i++)
        {
            double level = mean[i] * (1.0 + factor * (deviation[i] / 128.0 - 1.0));
            bool keep = (domain == null || domain[i] != 0) && values[i] < level;
            mask[i] = keep ? (byte)255 : (byte)0;
        }
    }

    static int DualScalar(byte[] gray, byte[]? domain, int[] labels, int width, int height, double limit, double strong, int minimum)
    {
        var allow = new byte[gray.Length];
        var peak = new byte[gray.Length];
        for (int i = 0; i < gray.Length; i++)
        {
            if (domain != null && domain[i] == 0)
                continue;
            if (gray[i] > 128.0 + limit || gray[i] < 128.0 - limit)
                allow[i] = 255;
            if (gray[i] > 128.0 + strong || gray[i] < 128.0 - strong)
                peak[i] = 255;
        }
        int label = 0;
        for (int i = 0; i < gray.Length; i++)
        {
            if (allow[i] == 0 || labels[i] != 0)
                continue;
            PaintLabel(labels, width, height, i, -1, allow);
            int area = 0;
            bool hot = false;
            for (int cursor = 0; cursor < labels.Length; cursor++)
            {
                if (labels[cursor] != -1)
                    continue;
                area++;
                if (peak[cursor] != 0)
                    hot = true;
            }
            if (area >= minimum && hot)
            {
                label++;
                for (int cursor = 0; cursor < labels.Length; cursor++)
                {
                    if (labels[cursor] == -1)
                        labels[cursor] = label;
                }
            }
            else
            {
                for (int cursor = 0; cursor < labels.Length; cursor++)
                {
                    if (labels[cursor] == -1)
                        labels[cursor] = -2;
                }
            }
        }
        return label;
    }

    static void PaintLabel(int[] labels, int width, int height, int start, int label, byte[] allow)
    {
        var queue = new Queue<int>();
        int[] drow = { -1, -1, -1, 0, 0, 1, 1, 1 };
        int[] dcol = { -1, 0, 1, -1, 1, -1, 0, 1 };
        labels[start] = label;
        queue.Enqueue(start);
        while (queue.Count > 0)
        {
            int index = queue.Dequeue();
            int row = index / width;
            int column = index - row * width;
            for (int neighbor = 0; neighbor < 8; neighbor++)
            {
                int nr = row + drow[neighbor];
                int nc = column + dcol[neighbor];
                if (nr < 0 || nc < 0 || nr >= height || nc >= width)
                    continue;
                int next = nr * width + nc;
                if (allow[next] == 0 || labels[next] != 0)
                    continue;
                labels[next] = label;
                queue.Enqueue(next);
            }
        }
    }

    static int GrowScalar(
        byte[] gray, byte[]? domain, int[] labels, int width, int height,
        int[] rows, int[] columns, double tolerance, int minimum)
    {
        var visited = new byte[gray.Length];
        int label = 0;
        int[] drow = { -1, -1, -1, 0, 0, 1, 1, 1 };
        int[] dcol = { -1, 0, 1, -1, 1, -1, 0, 1 };
        for (int seed = 0; seed < rows.Length; seed++)
        {
            int row = rows[seed];
            int column = columns[seed];
            if (row < 0 || column < 0 || row >= height || column >= width)
                continue;
            int start = row * width + column;
            if (visited[start] != 0 || (domain != null && domain[start] == 0))
                continue;
            var queue = new Queue<int>();
            var pixels = new List<int>();
            visited[start] = 1;
            queue.Enqueue(start);
            while (queue.Count > 0)
            {
                int index = queue.Dequeue();
                pixels.Add(index);
                int r = index / width;
                int c = index - r * width;
                for (int neighbor = 0; neighbor < 8; neighbor++)
                {
                    int nr = r + drow[neighbor];
                    int nc = c + dcol[neighbor];
                    if (nr < 0 || nc < 0 || nr >= height || nc >= width)
                        continue;
                    int next = nr * width + nc;
                    if (visited[next] != 0 || (domain != null && domain[next] == 0))
                        continue;
                    int delta = Math.Abs(gray[next] - gray[index]);
                    if (delta <= tolerance)
                    {
                        visited[next] = 1;
                        queue.Enqueue(next);
                    }
                }
            }
            if (pixels.Count < minimum)
                continue;
            label++;
            for (int i = 0; i < pixels.Count; i++)
                labels[pixels[i]] = label;
        }
        return label;
    }
}
