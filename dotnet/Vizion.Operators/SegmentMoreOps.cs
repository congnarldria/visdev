namespace Vizion;

public readonly struct HistoThresholds
{
    public HistoThresholds(double[] minThresh, double[] maxThresh)
    {
        MinThresh = minThresh;
        MaxThresh = maxThresh;
    }

    public double[] MinThresh { get; }
    public double[] MaxThresh { get; }
}

public static partial class Operators
{
    /// <summary>索引從 1 開始。區域取出物件，輪廓取出其中一條。可以一次取多個。</summary>
    /// <remarks>按索引取物件。</remarks>
    /// <example>
    /// <code>Vizion.select_obj (Objects, ObjectSelected, 1)</code>
    /// </example>
    /// <param name="objects">物件集合。</param>
    /// <param name="index">從 1 開始的索引。</param>
    /// <returns>挑出的物件。順序和索引相同。</returns>
    public static Region[] SelectObj(IReadOnlyList<Region> objects, double[] index)
    {
        if (objects == null)
            throw new ArgumentNullException(nameof(objects));
        if (index == null)
            throw new ArgumentNullException(nameof(index));
        var picked = new Region[index.Length];
        for (int i = 0; i < index.Length; i++)
        {
            int pos = (int)Math.Round(index[i]);
            if (pos < 1 || pos > objects.Count)
                throw new ArgumentException("物件索引 " + pos + " 超出範圍 1.." + objects.Count);
            picked[i] = objects[pos - 1];
        }
        return picked;
    }

    /// <summary>索引從 1 開始。可以一次取多條輪廓。</summary>
    /// <remarks>按索引取輪廓。</remarks>
    /// <example>
    /// <code>Vizion.select_obj (Objects, ObjectSelected, 1)</code>
    /// </example>
    /// <param name="contours">輪廓集合。</param>
    /// <param name="index">從 1 開始的索引。</param>
    /// <returns>挑出的輪廓。順序和索引相同。</returns>
    public static EdgePoint[][] SelectObj(EdgePoint[][] contours, double[] index)
    {
        if (contours == null)
            throw new ArgumentNullException(nameof(contours));
        if (index == null)
            throw new ArgumentNullException(nameof(index));
        var picked = new EdgePoint[index.Length][];
        for (int i = 0; i < index.Length; i++)
        {
            int pos = (int)Math.Round(index[i]);
            if (pos < 1 || pos > contours.Length)
                throw new ArgumentException("輪廓索引 " + pos + " 超出範圍 1.." + contours.Length);
            picked[i] = contours[pos - 1];
        }
        return picked;
    }

    /// <summary>只從給的種子開始。八鄰域裡，和當時區域平均的差不超過 Tolerance 就加入，平均會跟著新像素更新。</summary>
    /// <remarks>用變動平均做區域成長。</remarks>
    /// <example>
    /// <code>Vizion.regiongrowing_mean (Image, Regions, 120, 100, 10, 20)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="rows">種子列。</param>
    /// <param name="columns">種子行。</param>
    /// <param name="tolerance">和當時平均允許的灰階差。</param>
    /// <param name="minSize">小於這個面積就丟掉。</param>
    /// <returns>長出來的區域。每一塊一個區域。</returns>
    public static Region[] RegionGrowingMean(ImageBuffer image, double[] rows, double[] columns, double tolerance, int minSize)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (rows == null || columns == null)
            throw new ArgumentNullException(nameof(rows));
        int seeds = Math.Min(rows.Length, columns.Length);
        if (seeds == 0)
            throw new ArgumentException("至少要一個種子");
        var seedRows = new int[seeds];
        var seedColumns = new int[seeds];
        for (int i = 0; i < seeds; i++)
        {
            seedRows[i] = (int)Math.Round(rows[i]);
            seedColumns[i] = (int)Math.Round(columns[i]);
        }
        byte[] gray = image.GrayBytes();
        byte[]? domain = OwnDomain(image);
        var labels = new int[gray.Length];
        int count;
        if (!NativeCore.TryRegionGrowMean(gray, domain, labels, image.Width, image.Height, seedRows, seedColumns, tolerance, minSize, out count))
            count = GrowMeanScalar(gray, domain, labels, image.Width, image.Height, seedRows, seedColumns, tolerance, minSize);
        return SplitLabels(image.Width, image.Height, labels, count);
    }

    /// <summary>從左上角還沒被收下的像素開始。四鄰域裡，距離落在 MinTolerance 到 MaxTolerance 的像素併進來。</summary>
    /// <remarks>依相鄰像素的距離分割。灰階用絕對差，彩色用三通道的歐氏距離。</remarks>
    /// <example>
    /// <code>Vizion.regiongrowing_n (Image, Regions, 'euclid', 0, 5, 1)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="metric">距離。這版只做 euclid。</param>
    /// <param name="minTolerance">距離下限。</param>
    /// <param name="maxTolerance">距離上限。</param>
    /// <param name="minSize">小於這個面積就丟掉。</param>
    /// <returns>長出來的區域。每一塊一個區域。</returns>
    public static Region[] RegionGrowingN(ImageBuffer image, string metric, double minTolerance, double maxTolerance, int minSize)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (Clean(metric) != "euclid")
            throw new ArgumentException("Metric 這版只做 euclid");
        if (minTolerance < 0 || maxTolerance < minTolerance)
            throw new ArgumentException("容差要是 0 以上，而且 MaxTolerance 不能小於 MinTolerance");
        var labels = new int[image.Width * image.Height];
        int count;
        if (!NativeCore.TryRegionGrowN(image.Pixels, labels, image.Width, image.Height, image.Channels, minTolerance, maxTolerance, minSize, out count))
            count = GrowNScalar(image.Pixels, labels, image.Width, image.Height, image.Channels, minTolerance, maxTolerance, minSize);
        return SplitLabels(image.Width, image.Height, labels, count);
    }

    /// <summary>先把直方圖做高斯平滑，再以波谷切開。沒有波谷時就是整段。空的區間會丟掉。</summary>
    /// <remarks>從直方圖找出灰階區間。</remarks>
    /// <example>
    /// <code>Vizion.histo_to_thresh (Histogram, 2, MinThresh, MaxThresh)</code>
    /// </example>
    /// <param name="histogram">灰階直方圖。</param>
    /// <param name="sigma">平滑的標準差，單位是灰階格。0 不平滑。</param>
    /// <returns>每一段的下限和上限。</returns>
    public static HistoThresholds HistoToThresh(double[] histogram, double sigma)
    {
        if (histogram == null)
            throw new ArgumentNullException(nameof(histogram));
        if (histogram.Length < 3)
            throw new ArgumentException("直方圖至少要 3 格");
        if (sigma < 0)
            throw new ArgumentException("Sigma 不能是負的");
        var lows = new int[histogram.Length];
        var highs = new int[histogram.Length];
        int segments;
        if (!NativeCore.TryHistoToThresh(histogram, sigma, lows, highs, out segments))
            segments = ValleyScalar(histogram, sigma, lows, highs);
        var minThresh = new double[segments];
        var maxThresh = new double[segments];
        for (int i = 0; i < segments; i++)
        {
            minThresh[i] = lows[i];
            maxThresh[i] = highs[i];
        }
        return new HistoThresholds(minThresh, maxThresh);
    }

    /// <summary>先做灰階直方圖，用 Sigma 平滑後在谷底切開。每個有像素的區間變成一個物件，暗的在前面。</summary>
    /// <remarks>依直方圖的谷底分割。彩色先轉成亮度。</remarks>
    /// <example>
    /// <code>Vizion.auto_threshold (Image, Regions, 2)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="sigma">直方圖平滑尺度。0 不平滑。</param>
    /// <returns>每個灰階區間一個區域。</returns>
    public static Region[] AutoThreshold(ImageBuffer image, double sigma)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (sigma < 0)
            throw new ArgumentException("Sigma 不能是負的");
        byte[] gray = image.GrayBytes();
        byte[]? domain = OwnDomain(image);
        var labels = new int[gray.Length];
        int count;
        if (!NativeCore.TryAutoThreshold(gray, domain, labels, sigma, out count))
            count = AutoScalar(gray, domain, labels, sigma);
        return SplitLabels(image.Width, image.Height, labels, count);
    }

    /// <summary>先找最高的背景波峰，再找另一側的字元波峰，門檻放在兩個波峰之間。Percent 100 就用波谷。</summary>
    /// <remarks>用直方圖把字元分開。找不到第二個波峰時改用 Otsu。直方圖只用 HistoRegion，分類套到整張圖。</remarks>
    /// <example>
    /// <code>Vizion.char_threshold (Image, HistoRegion, Characters, 2, 90, Threshold)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="histoRegion">用來統計的區域。</param>
    /// <param name="sigma">直方圖平滑。</param>
    /// <param name="percent">0 到 100。100 最靠近波谷。</param>
    /// <param name="threshold">實際用的門檻。</param>
    /// <returns>字元區域。有定義域時只留定義域裡的像素。</returns>
    public static Region CharThreshold(ImageBuffer image, Region histoRegion, double sigma, double percent, out int threshold)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (histoRegion == null)
            throw new ArgumentNullException(nameof(histoRegion));
        if (image.Width != histoRegion.Width || image.Height != histoRegion.Height)
            throw new ArgumentException("影像和區域的尺寸不同");
        if (!(sigma > 0))
            throw new ArgumentException("Sigma 必須大於 0");
        if (percent < 0 || percent > 100)
            throw new ArgumentException("Percent 要在 0 到 100");
        byte[] gray = image.GrayBytes();
        byte[] histo = MaskOf(histoRegion);
        if (CountBytes(histo) == 0)
            throw new ArgumentException("HistoRegion 是空的");
        byte[]? domain = OwnDomain(image);
        var mask = new byte[gray.Length];
        if (!NativeCore.TryCharThreshold(gray, histo, domain, mask, sigma, percent, out threshold))
            threshold = CharScalar(gray, histo, domain, mask, sigma, percent);
        return RunsOf(image.Width, image.Height, mask);
    }

    /// <summary>max_separability 用 Otsu，smooth_histo 用平滑後的波谷，triangle 用三角形法。local_mean 和 local_gauss 用 31 像素鄰域減 5。</summary>
    /// <remarks>自動二值化。light 保留亮處，dark 保留暗處。彩色先轉成亮度。</remarks>
    /// <example>
    /// <code>Vizion.binary_threshold (Image, Region, 'max_separability', 'light', UsedThreshold)</code>
    /// </example>
    /// <param name="image">灰階或彩色影像。</param>
    /// <param name="method">max_separability、smooth_histo、triangle、local_mean 或 local_gauss。</param>
    /// <param name="lightDark">light 或 dark。</param>
    /// <param name="usedThreshold">實際使用的門檻。局部方法是 5。</param>
    /// <returns>二值化區域。有定義域時只留定義域裡的像素。</returns>
    public static Region BinaryThreshold(ImageBuffer image, string method, string lightDark, out int usedThreshold)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        int code = BinaryCodeMore(method);
        int dark = DarkCode(lightDark);
        byte[] gray = image.GrayBytes();
        byte[]? domain = OwnDomain(image);
        var mask = new byte[gray.Length];
        if (!NativeCore.TryBinaryThreshold(gray, domain, mask, image.Width, image.Height, code, dark, out usedThreshold))
            usedThreshold = BinaryScalar(gray, domain, mask, image.Width, image.Height, code, dark);
        return RunsOf(image.Width, image.Height, mask);
    }

    /// <summary>用橢圓頂帽或黑帽把小點從背景分離，再靠 PixelShift 門檻。直徑會變成奇數，範圍 3 到 31。</summary>
    /// <remarks>找出小亮點或小暗點。彩色先轉成亮度。</remarks>
    /// <example>
    /// <code>Vizion.dots_image (Image, DotRegion, 5, 'light', 30)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="diameter">點的大約直徑。</param>
    /// <param name="filterType">light、dark 或 all。</param>
    /// <param name="pixelShift">要比周圍高或低幾個灰階。</param>
    /// <returns>點狀區域。有定義域時只留定義域裡的像素。</returns>
    public static Region DotsImage(ImageBuffer image, int diameter, string filterType, double pixelShift)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        int size = diameter | 1;
        if (size < 3 || size > 31)
            throw new ArgumentException("Diameter 要在 3 到 31");
        int kind = DotKind(filterType);
        if (pixelShift < 0)
            throw new ArgumentException("PixelShift 不能是負的");
        byte[] gray = image.GrayBytes();
        byte[]? domain = OwnDomain(image);
        var mask = new byte[gray.Length];
        if (!NativeCore.TryDotsImage(gray, domain, mask, image.Width, image.Height, size, kind, pixelShift))
            DotsScalar(gray, domain, mask, image.Width, image.Height, size, kind, pixelShift);
        return RunsOf(image.Width, image.Height, mask);
    }

    static int GrowMeanScalar(
        byte[] gray, byte[]? domain, int[] labels, int width, int height, int[] rows, int[] columns, double tolerance, int minSize)
    {
        int[] drow = { -1, -1, -1, 0, 0, 1, 1, 1 };
        int[] dcol = { -1, 0, 1, -1, 1, -1, 0, 1 };
        var visited = new byte[gray.Length];
        var queue = new int[gray.Length];
        int label = 0;
        Array.Clear(labels, 0, labels.Length);
        for (int seed = 0; seed < rows.Length; seed++)
        {
            int row = rows[seed];
            int column = columns[seed];
            if ((uint)row >= (uint)height || (uint)column >= (uint)width)
                continue;
            int start = row * width + column;
            if (visited[start] != 0 || (domain != null && domain[start] == 0))
                continue;
            int head = 0;
            int tail = 0;
            visited[start] = 1;
            queue[tail++] = start;
            long total = gray[start];
            int area = 1;
            while (head < tail)
            {
                int index = queue[head++];
                int y = index / width;
                int x = index - y * width;
                double mean = (double)total / area;
                for (int neighbor = 0; neighbor < 8; neighbor++)
                {
                    int nextRow = y + drow[neighbor];
                    int nextColumn = x + dcol[neighbor];
                    if ((uint)nextRow >= (uint)height || (uint)nextColumn >= (uint)width)
                        continue;
                    int next = nextRow * width + nextColumn;
                    if (visited[next] != 0 || (domain != null && domain[next] == 0))
                        continue;
                    double delta = gray[next] - mean;
                    if (delta < 0)
                        delta = -delta;
                    if (delta <= tolerance)
                    {
                        visited[next] = 1;
                        queue[tail++] = next;
                        total += gray[next];
                        area++;
                    }
                }
            }
            if (area >= minSize)
            {
                label++;
                for (int cursor = 0; cursor < tail; cursor++)
                    labels[queue[cursor]] = label;
            }
        }
        return label;
    }

    static int GrowNScalar(byte[] pixels, int[] labels, int width, int height, int channels, double low, double high, int minSize)
    {
        int[] drow = { -1, 1, 0, 0 };
        int[] dcol = { 0, 0, -1, 1 };
        int count = width * height;
        var visited = new byte[count];
        var queue = new int[count];
        int label = 0;
        Array.Clear(labels, 0, labels.Length);
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                int start = row * width + column;
                if (visited[start] != 0)
                    continue;
                int head = 0;
                int tail = 0;
                visited[start] = 1;
                queue[tail++] = start;
                while (head < tail)
                {
                    int index = queue[head++];
                    int y = index / width;
                    int x = index - y * width;
                    for (int neighbor = 0; neighbor < 4; neighbor++)
                    {
                        int nextRow = y + drow[neighbor];
                        int nextColumn = x + dcol[neighbor];
                        if ((uint)nextRow >= (uint)height || (uint)nextColumn >= (uint)width)
                            continue;
                        int next = nextRow * width + nextColumn;
                        if (visited[next] != 0)
                            continue;
                        if (low <= Gap(pixels, channels, next, index) && Gap(pixels, channels, next, index) <= high)
                        {
                            visited[next] = 1;
                            queue[tail++] = next;
                        }
                    }
                }
                if (tail >= minSize)
                {
                    label++;
                    for (int cursor = 0; cursor < tail; cursor++)
                        labels[queue[cursor]] = label;
                }
            }
        }
        return label;
    }

    static double Gap(byte[] pixels, int channels, int left, int right)
    {
        if (channels == 1)
            return Math.Abs(pixels[left] - pixels[right]);
        double sum = 0;
        for (int channel = 0; channel < channels; channel++)
        {
            double delta = pixels[left * channels + channel] - pixels[right * channels + channel];
            sum += delta * delta;
        }
        return Math.Sqrt(sum);
    }

    static int ValleyScalar(double[] counts, double sigma, int[] lows, int[] highs)
    {
        var smooth = new double[counts.Length];
        if (sigma > 0)
        {
            int size = OddSpan(counts.Length, sigma);
            double[] kernel = SpanKernel(size, sigma);
            BlurReflect(counts, kernel, smooth);
        }
        else
        {
            Array.Copy(counts, smooth, counts.Length);
        }
        var valleys = new List<int>();
        int index = 1;
        while (index < smooth.Length - 1)
        {
            if (smooth[index] < smooth[index - 1])
            {
                int end = index;
                while (end + 1 < smooth.Length && smooth[end + 1] <= smooth[end])
                    end++;
                if (end + 1 < smooth.Length && smooth[end + 1] > smooth[end])
                    valleys.Add((index + end) / 2);
                index = end + 1;
            }
            else
            {
                index++;
            }
        }
        int kept = 0;
        for (int band = 0; band <= valleys.Count; band++)
        {
            int start = band == 0 ? 0 : valleys[band - 1] + 1;
            int end = band == valleys.Count ? counts.Length - 1 : valleys[band];
            if (end < start)
                continue;
            double total = 0;
            for (int bin = start; bin <= end; bin++)
                total += counts[bin];
            if (total <= 0)
                continue;
            lows[kept] = start;
            highs[kept] = end;
            kept++;
        }
        if (kept == 0)
        {
            lows[0] = 0;
            highs[0] = counts.Length - 1;
            kept = 1;
        }
        return kept;
    }

    static int AutoScalar(byte[] gray, byte[]? domain, int[] labels, double sigma)
    {
        var hist = new double[256];
        int samples = 0;
        Array.Clear(labels, 0, labels.Length);
        for (int i = 0; i < gray.Length; i++)
        {
            if (domain != null && domain[i] == 0)
                continue;
            hist[gray[i]] += 1;
            samples++;
        }
        if (samples == 0)
            return 0;
        var lows = new int[256];
        var highs = new int[256];
        int kept = ValleyScalar(hist, sigma, lows, highs);
        for (int band = 0; band < kept; band++)
        {
            for (int i = 0; i < gray.Length; i++)
            {
                if (domain != null && domain[i] == 0)
                    continue;
                if (gray[i] >= lows[band] && gray[i] <= highs[band])
                    labels[i] = band + 1;
            }
        }
        return kept;
    }

    static int CharScalar(byte[] gray, byte[] histo, byte[]? domain, byte[] mask, double sigma, double percent)
    {
        var hist = new double[256];
        for (int i = 0; i < gray.Length; i++)
        {
            if (histo[i] != 0)
                hist[gray[i]] += 1;
        }
        int radius = (int)Math.Round(sigma * 3.0);
        if (radius < 1)
            radius = 1;
        double[] kernel = SpanKernel(radius * 2 + 1, sigma);
        var smooth = new double[256];
        ConvolveSame(hist, kernel, smooth);
        int peak = 0;
        for (int i = 1; i < 256; i++)
        {
            if (smooth[i] > smooth[peak])
                peak = i;
        }
        int secondary = -1;
        double best = 0;
        double floor = smooth[peak] * 0.01;
        for (int i = 1; i < 255; i++)
        {
            if (Math.Abs(i - peak) < 8)
                continue;
            if (smooth[i] > smooth[i - 1] && smooth[i] > smooth[i + 1] && smooth[i] > best && smooth[i] > floor)
            {
                best = smooth[i];
                secondary = i;
            }
        }
        int level;
        bool darkSide;
        if (secondary < 0)
        {
            level = OtsuScalar(gray);
            int dark = 0;
            for (int i = 0; i < gray.Length; i++)
            {
                if (gray[i] <= level)
                    dark++;
            }
            darkSide = dark <= gray.Length - dark;
        }
        else
        {
            int low = Math.Min(peak, secondary);
            int high = Math.Max(peak, secondary);
            int valley = low;
            for (int i = low + 1; i <= high; i++)
            {
                if (smooth[i] < smooth[valley])
                    valley = i;
            }
            level = (int)Math.Round(valley + (peak - valley) * (100.0 - percent) / 100.0);
            if (level < 0)
                level = 0;
            if (level > 255)
                level = 255;
            darkSide = secondary < peak;
        }
        for (int i = 0; i < gray.Length; i++)
        {
            bool keep = darkSide ? gray[i] <= level : gray[i] >= level;
            if (domain != null && domain[i] == 0)
                keep = false;
            mask[i] = keep ? (byte)255 : (byte)0;
        }
        return level;
    }

    static int BinaryScalar(byte[] gray, byte[]? domain, byte[] mask, int width, int height, int method, int dark)
    {
        int level;
        if (method <= 2)
        {
            level = method == 0 ? OtsuScalar(gray) : method == 1 ? SmoothHistoScalar(gray) : TriangleScalar(gray);
            for (int i = 0; i < gray.Length; i++)
                mask[i] = gray[i] > level ? (byte)255 : (byte)0;
        }
        else
        {
            byte[] local = new byte[gray.Length];
            if (method == 3)
                BoxMean(gray, local, width, height, 31);
            else
                GaussMean(gray, local, width, height, 31);
            for (int i = 0; i < gray.Length; i++)
                mask[i] = gray[i] + 5 > local[i] ? (byte)255 : (byte)0;
            level = 5;
        }
        for (int i = 0; i < gray.Length; i++)
        {
            if (dark != 0)
                mask[i] = mask[i] == 0 ? (byte)255 : (byte)0;
            if (domain != null && domain[i] == 0)
                mask[i] = 0;
        }
        return level;
    }

    static void DotsScalar(byte[] gray, byte[]? domain, byte[] mask, int width, int height, int diameter, int kind, double shift)
    {
        int radius = diameter / 2;
        Array.Clear(mask, 0, mask.Length);
        if (kind == 0 || kind == 2)
        {
            byte[] opened = OpenEllipse(gray, width, height, radius);
            for (int i = 0; i < gray.Length; i++)
            {
                int diff = gray[i] - opened[i];
                if (diff < 0)
                    diff = 0;
                if (diff >= shift)
                    mask[i] = 255;
            }
        }
        if (kind == 1 || kind == 2)
        {
            byte[] closed = CloseEllipse(gray, width, height, radius);
            for (int i = 0; i < gray.Length; i++)
            {
                int diff = closed[i] - gray[i];
                if (diff < 0)
                    diff = 0;
                if (diff >= shift)
                    mask[i] = 255;
            }
        }
        if (domain != null)
        {
            for (int i = 0; i < mask.Length; i++)
            {
                if (domain[i] == 0)
                    mask[i] = 0;
            }
        }
    }

    static byte[] OpenEllipse(byte[] gray, int width, int height, int radius)
    {
        byte[] eroded = FilterOffsets(gray, width, height, EllipseOffsets(radius), true);
        return FilterOffsets(eroded, width, height, EllipseOffsets(radius), false);
    }

    static byte[] CloseEllipse(byte[] gray, int width, int height, int radius)
    {
        byte[] dilated = FilterOffsets(gray, width, height, EllipseOffsets(radius), false);
        return FilterOffsets(dilated, width, height, EllipseOffsets(radius), true);
    }

    static int OtsuScalar(byte[] gray)
    {
        var hist = new double[256];
        for (int i = 0; i < gray.Length; i++)
            hist[gray[i]] += 1;
        double sumAll = 0;
        for (int i = 0; i < 256; i++)
            sumAll += i * hist[i];
        double sumBack = 0;
        double weightBack = 0;
        double best = -1;
        int threshold = 0;
        for (int i = 0; i < 256; i++)
        {
            weightBack += hist[i];
            if (weightBack == 0)
                continue;
            double weightFore = gray.Length - weightBack;
            if (weightFore == 0)
                break;
            sumBack += i * hist[i];
            double meanBack = sumBack / weightBack;
            double meanFore = (sumAll - sumBack) / weightFore;
            double between = weightBack * weightFore * (meanBack - meanFore) * (meanBack - meanFore);
            if (between > best)
            {
                best = between;
                threshold = i;
            }
        }
        return threshold;
    }

    static int SmoothHistoScalar(byte[] gray)
    {
        var hist = new double[256];
        for (int i = 0; i < gray.Length; i++)
            hist[gray[i]] += 1;
        double[] kernel = SpanKernel(15, 2);
        var smooth = new double[256];
        ConvolveSame(hist, kernel, smooth);
        var peaks = new List<int>();
        for (int i = 1; i < 255; i++)
        {
            if (smooth[i] >= smooth[i - 1] && smooth[i] >= smooth[i + 1] && smooth[i] > 0)
                peaks.Add(i);
        }
        if (peaks.Count < 2)
            return OtsuScalar(gray);
        peaks.Sort((left, right) =>
        {
            int byValue = smooth[right].CompareTo(smooth[left]);
            return byValue != 0 ? byValue : left.CompareTo(right);
        });
        int low = Math.Min(peaks[0], peaks[1]);
        int high = Math.Max(peaks[0], peaks[1]);
        int valley = low;
        for (int i = low + 1; i <= high; i++)
        {
            if (smooth[i] < smooth[valley])
                valley = i;
        }
        return valley;
    }

    static int TriangleScalar(byte[] gray)
    {
        var hist = new int[256];
        for (int i = 0; i < gray.Length; i++)
            hist[gray[i]]++;
        int left = 0;
        for (int i = 0; i < 256; i++)
        {
            if (hist[i] > 0)
            {
                left = i;
                break;
            }
        }
        if (left > 0)
            left--;
        int right = 0;
        for (int i = 255; i > 0; i--)
        {
            if (hist[i] > 0)
            {
                right = i;
                break;
            }
        }
        if (right < 255)
            right++;
        int peak = 0;
        int peakCount = 0;
        for (int i = 0; i < 256; i++)
        {
            if (hist[i] > peakCount)
            {
                peakCount = hist[i];
                peak = i;
            }
        }
        bool flipped = false;
        if (peak - left < right - peak)
        {
            flipped = true;
            Array.Reverse(hist);
            left = 255 - right;
            peak = 255 - peak;
        }
        double thresh = left;
        double best = 0;
        double height = peakCount;
        double slope = left - peak;
        for (int i = left + 1; i <= peak; i++)
        {
            double distance = height * i + slope * hist[i];
            if (distance > best)
            {
                best = distance;
                thresh = i;
            }
        }
        thresh -= 1;
        if (flipped)
            thresh = 255 - thresh;
        return (int)thresh;
    }

    static void BoxMean(byte[] source, byte[] mean, int width, int height, int side)
    {
        int radius = side / 2;
        int paddedWidth = width + radius * 2;
        int paddedHeight = height + radius * 2;
        int stride = paddedWidth + 1;
        var padded = new byte[paddedWidth * paddedHeight];
        var integral = new long[(paddedHeight + 1) * stride];
        for (int row = 0; row < paddedHeight; row++)
        {
            int sourceRow = ClampIndex(row - radius, height);
            for (int column = 0; column < paddedWidth; column++)
            {
                int sourceColumn = ClampIndex(column - radius, width);
                padded[row * paddedWidth + column] = source[sourceRow * width + sourceColumn];
            }
        }
        for (int row = 0; row < paddedHeight; row++)
        {
            for (int column = 0; column < paddedWidth; column++)
            {
                long sum = padded[row * paddedWidth + column];
                sum += integral[row * stride + column + 1];
                sum += integral[(row + 1) * stride + column];
                sum -= integral[row * stride + column];
                integral[(row + 1) * stride + column + 1] = sum;
            }
        }
        double area = (double)side * side;
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                long sum = integral[(row + side) * stride + column + side];
                sum -= integral[row * stride + column + side];
                sum -= integral[(row + side) * stride + column];
                sum += integral[row * stride + column];
                mean[row * width + column] = (byte)(int)Math.Round(sum / area);
            }
        }
    }

    static void GaussMean(byte[] source, byte[] mean, int width, int height, int side)
    {
        double[] exact = SpanKernel(side, 0);
        var kernel = new float[side];
        for (int i = 0; i < side; i++)
            kernel[i] = (float)exact[i];
        int radius = side / 2;
        var horizontal = new float[width * height];
        var vertical = new float[width * height];
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                float sum = 0;
                for (int tap = 0; tap < side; tap++)
                {
                    int sample = ClampIndex(column + tap - radius, width);
                    sum = sum + source[row * width + sample] * kernel[tap];
                }
                horizontal[row * width + column] = sum;
            }
        }
        for (int column = 0; column < width; column++)
        {
            for (int row = 0; row < height; row++)
            {
                float sum = 0;
                for (int tap = 0; tap < side; tap++)
                {
                    int sample = ClampIndex(row + tap - radius, height);
                    sum = sum + horizontal[sample * width + column] * kernel[tap];
                }
                vertical[row * width + column] = sum;
            }
        }
        for (int i = 0; i < vertical.Length; i++)
        {
            int rounded = (int)Math.Round((double)vertical[i]);
            if (rounded < 0)
                rounded = 0;
            if (rounded > 255)
                rounded = 255;
            mean[i] = (byte)rounded;
        }
    }

    static double[] SpanKernel(int size, double sigma)
    {
        double sigmaX = sigma > 0 ? sigma : ((size - 1) * 0.5 - 1.0) * 0.3 + 0.8;
        double scale = -0.5 / (sigmaX * sigmaX);
        var kernel = new double[size];
        double sum = 0;
        for (int i = 0; i < size; i++)
        {
            double offset = i - (size - 1) * 0.5;
            kernel[i] = Math.Exp(scale * offset * offset);
            sum += kernel[i];
        }
        for (int i = 0; i < size; i++)
            kernel[i] /= sum;
        return kernel;
    }

    static void ConvolveSame(double[] signal, double[] kernel, double[] destination)
    {
        int radius = kernel.Length / 2;
        for (int index = 0; index < signal.Length; index++)
        {
            double sum = 0;
            for (int tap = 0; tap < kernel.Length; tap++)
            {
                int sample = index + tap - radius;
                if (sample >= 0 && sample < signal.Length)
                    sum += signal[sample] * kernel[tap];
            }
            destination[index] = sum;
        }
    }

    static void BlurReflect(double[] signal, double[] kernel, double[] destination)
    {
        int radius = kernel.Length / 2;
        for (int index = 0; index < signal.Length; index++)
        {
            double sum = 0;
            for (int tap = 0; tap < kernel.Length; tap++)
                sum += signal[HistoReflect(index + tap - radius, signal.Length)] * kernel[tap];
            destination[index] = sum;
        }
    }

    static int HistoReflect(int position, int length)
    {
        if (length <= 1)
            return 0;
        while (position < 0 || position >= length)
        {
            if (position < 0)
                position = -position;
            if (position >= length)
                position = length + length - position - 2;
        }
        return position;
    }

    static int OddSpan(int length, double sigma)
    {
        int raw = (int)Math.Round(sigma * 6.0);
        int odd = raw | 1;
        int cap = (length & 1) == 1 ? length : length - 1;
        int size = odd < 3 ? 3 : odd;
        if (size > cap)
            size = cap;
        return size;
    }

    static int ClampIndex(int position, int length)
    {
        if (position < 0)
            return 0;
        if (position >= length)
            return length - 1;
        return position;
    }

    static byte[] MaskOf(Region region)
    {
        var mask = new byte[region.Width * region.Height];
        for (int i = 0; i < region.Rows.Length; i++)
        {
            int row = region.Rows[i];
            int column = region.Columns[i];
            int length = region.Lengths[i];
            if ((uint)row >= (uint)region.Height || length <= 0)
                continue;
            int offset = row * region.Width + column;
            for (int step = 0; step < length && column + step < region.Width; step++)
                mask[offset + step] = 255;
        }
        return mask;
    }

    static int CountBytes(byte[] values)
    {
        int count = 0;
        for (int i = 0; i < values.Length; i++)
        {
            if (values[i] != 0)
                count++;
        }
        return count;
    }

    static int BinaryCodeMore(string method)
    {
        string name = Clean(method);
        if (name == "max_separability" || name == "otsu")
            return 0;
        if (name == "smooth_histo")
            return 1;
        if (name == "triangle")
            return 2;
        if (name == "local_mean")
            return 3;
        if (name == "local_gauss")
            return 4;
        throw new ArgumentException("Method 可以是 max_separability、smooth_histo、triangle、local_mean、local_gauss");
    }

    static int DarkCode(string mode)
    {
        string name = Clean(mode);
        if (name == "light")
            return 0;
        if (name == "dark")
            return 1;
        throw new ArgumentException("LightDark 只能是 light 或 dark");
    }

    static int DotKind(string filter)
    {
        string name = Clean(filter);
        if (name == "light")
            return 0;
        if (name == "dark")
            return 1;
        if (name == "all")
            return 2;
        throw new ArgumentException("FilterType 只能是 light、dark 或 all");
    }

    static string Clean(string text)
    {
        if (text == null)
            throw new ArgumentNullException(nameof(text));
        return text.Trim().Trim('\'', '"').ToLowerInvariant();
    }
}
