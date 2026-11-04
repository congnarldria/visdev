namespace Vizion;

public static partial class Operators
{
    /// <summary>3×3 裡比周圍高的像素。比遮罩還大的平坦高台，內部不會被標出來。彩色先轉成亮度。</summary>
    /// <remarks>局部極大。不填 MinGray 時用 0。</remarks>
    /// <example>
    /// <code>Vizion.local_max (Image, LocalMaxima, 1)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="minGray">最低灰階。</param>
    /// <returns>局部極大。有定義域時只留定義域裡的像素。</returns>
    public static Region LocalMax(ImageBuffer image, double minGray)
    {
        return Peak(image, minGray, minima: false);
    }

    /// <summary>3×3 裡比周圍暗的像素。比遮罩還大的平坦暗區，內部不會被標出來。彩色先轉成亮度。</summary>
    /// <remarks>局部極小。不填 MaxGray 時用 255。</remarks>
    /// <example>
    /// <code>Vizion.local_min (Image, LocalMinima, 255)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="maxGray">最高灰階。</param>
    /// <returns>局部極小。有定義域時只留定義域裡的像素。</returns>
    public static Region LocalMin(ImageBuffer image, double maxGray)
    {
        return Peak(image, maxGray, minima: true);
    }

    /// <summary>abs 是兩邊相差超過門檻。light 是比參考亮，dark 是比參考暗。彩色先轉成亮度。有定義域時只比定義域裡面。</summary>
    /// <remarks>找出和參考影像不一樣的地方。</remarks>
    /// <example>
    /// <code>Vizion.check_difference (Image, Pattern, Region, 'abs', 20)</code>
    /// </example>
    /// <param name="image">要比對的影像。</param>
    /// <param name="pattern">參考影像。</param>
    /// <param name="mode">abs、light 或 dark。</param>
    /// <param name="threshold">超過這個灰階差才留下。</param>
    /// <returns>差太多的像素。</returns>
    public static Region CheckDifference(ImageBuffer image, ImageBuffer pattern, string mode, double threshold)
    {
        if (image == null || pattern == null)
            throw new ArgumentNullException(nameof(image));
        if (image.Width != pattern.Width || image.Height != pattern.Height)
            throw new ArgumentException("兩張影像的寬高要相同");
        if (threshold < 0)
            throw new ArgumentException("Threshold 不能是負的");
        int code = DiffCode(mode);
        byte[] left = image.GrayBytes();
        byte[] right = pattern.GrayBytes();
        var mask = new byte[left.Length];
        if (!NativeCore.TryCheckDiff(left, right, mask, threshold, code))
            CheckDiffScalar(left, right, mask, threshold, code);
        Region found = RunsOf(image.Width, image.Height, mask);
        if (image.Domain != null)
            found = Intersection(found, image.Domain);
        if (pattern.Domain != null)
            found = Intersection(found, pattern.Domain);
        return found;
    }

    /// <summary>灰階 128 當作 0。相鄰像素一正一負，或其中一個剛好是 128 時，留下離 0 較近的那一側。彩色先轉成亮度。</summary>
    /// <remarks>灰階過零。</remarks>
    /// <example>
    /// <code>Vizion.zero_crossing (Image, RegionCrossing)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <returns>過零的位置。有定義域時只留定義域裡的像素。</returns>
    public static Region ZeroCrossing(ImageBuffer image)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        byte[] gray = image.GrayBytes();
        var mask = new byte[gray.Length];
        if (!NativeCore.TryZeroCross(gray, mask, image.Width, image.Height))
            ZeroScalar(gray, mask, image.Width, image.Height);
        Region found = RunsOf(image.Width, image.Height, mask);
        if (image.Domain != null)
            found = Intersection(found, image.Domain);
        return found;
    }

    /// <summary>先留下 MinGray 到 MaxGray 的像素。同一列上比 MinSize 短的連續段會丟掉。彩色先轉成亮度。</summary>
    /// <remarks>留下夠長的灰階段。</remarks>
    /// <example>
    /// <code>Vizion.fast_threshold (Image, Region, 200, 255, 3)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="minGray">最低灰階。</param>
    /// <param name="maxGray">最高灰階。</param>
    /// <param name="minSize">同一列至少要連續幾個像素。</param>
    /// <returns>夠長的亮段。</returns>
    public static Region FastThreshold(ImageBuffer image, int minGray, int maxGray, int minSize)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (minGray > maxGray)
            throw new ArgumentException("MinGray 不能大於 MaxGray");
        if (minSize < 1)
            throw new ArgumentException("MinSize 至少要 1");
        byte[] gray = image.GrayBytes();
        var mask = new byte[gray.Length];
        if (!NativeCore.TryFastThreshold(gray, mask, image.Width, image.Height, minGray, maxGray, minSize))
            FastScalar(gray, mask, image.Width, image.Height, minGray, maxGray, minSize);
        return RunsOf(image.Width, image.Height, mask);
    }

    static Region Peak(ImageBuffer image, double level, bool minima)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        byte[] gray = image.GrayBytes();
        var mask = new byte[gray.Length];
        if (!NativeCore.TryLocalPeak(gray, mask, image.Width, image.Height, level, minima))
            PeakScalar(gray, mask, image.Width, image.Height, level, minima);
        Region found = RunsOf(image.Width, image.Height, mask);
        if (image.Domain != null)
            found = Intersection(found, image.Domain);
        return found;
    }

    static int DiffCode(string mode)
    {
        if (mode == null)
            throw new ArgumentNullException(nameof(mode));
        string name = mode.Trim().Trim('\'').Trim('"').ToLowerInvariant();
        if (name == "abs")
            return 0;
        if (name == "light")
            return 1;
        if (name == "dark")
            return 2;
        throw new ArgumentException("Mode 只能是 abs、light 或 dark");
    }

    static Region RunsOf(int width, int height, byte[] mask)
    {
        var rows = new List<int>();
        var columns = new List<int>();
        var lengths = new List<int>();
        for (int row = 0; row < height; row++)
        {
            int column = 0;
            while (column < width)
            {
                if (mask[row * width + column] == 0)
                {
                    column++;
                    continue;
                }
                int start = column;
                column++;
                while (column < width && mask[row * width + column] != 0)
                    column++;
                rows.Add(row);
                columns.Add(start);
                lengths.Add(column - start);
            }
        }
        return new Region(width, height, rows.ToArray(), columns.ToArray(), lengths.ToArray());
    }

    static int SampleAt(byte[] gray, int width, int height, int row, int column, int outside)
    {
        if (row < 0 || column < 0 || row >= height || column >= width)
            return outside;
        return gray[row * width + column];
    }

    static void PeakScalar(byte[] gray, byte[] mask, int width, int height, double level, bool minima)
    {
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                int value = gray[row * width + column];
                int highest = 0;
                int lowest = 255;
                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int highSample = SampleAt(gray, width, height, row + dy, column + dx, 0);
                        int lowSample = SampleAt(gray, width, height, row + dy, column + dx, 255);
                        if (highSample > highest)
                            highest = highSample;
                        if (lowSample < lowest)
                            lowest = lowSample;
                    }
                }
                bool keep = !minima
                    ? value == highest && value > lowest && (!(level > 0) || value >= level)
                    : value == lowest && value < highest && (!(level < 255) || value <= level);
                mask[row * width + column] = keep ? (byte)255 : (byte)0;
            }
        }
    }

    static void CheckDiffScalar(byte[] image, byte[] pattern, byte[] mask, double limit, int mode)
    {
        for (int i = 0; i < image.Length; i++)
        {
            int delta = image[i] - pattern[i];
            bool keep = mode == 1 ? delta > limit : mode == 2 ? -delta > limit : Math.Abs(delta) > limit;
            mask[i] = keep ? (byte)255 : (byte)0;
        }
    }

    static void ZeroScalar(byte[] gray, byte[] mask, int width, int height)
    {
        Array.Clear(mask, 0, mask.Length);
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width - 1; column++)
            {
                int left = gray[row * width + column] - 128;
                int right = gray[row * width + column + 1] - 128;
                bool change = left * right < 0 || ((left == 0) ^ (right == 0));
                int leftAbs = Math.Abs(left);
                int rightAbs = Math.Abs(right);
                if (change && leftAbs <= rightAbs)
                    mask[row * width + column] = 255;
                if (change && rightAbs <= leftAbs)
                    mask[row * width + column + 1] = 255;
            }
        }
        for (int row = 0; row < height - 1; row++)
        {
            for (int column = 0; column < width; column++)
            {
                int up = gray[row * width + column] - 128;
                int down = gray[(row + 1) * width + column] - 128;
                bool change = up * down < 0 || ((up == 0) ^ (down == 0));
                int upAbs = Math.Abs(up);
                int downAbs = Math.Abs(down);
                if (change && upAbs <= downAbs)
                    mask[row * width + column] = 255;
                if (change && downAbs <= upAbs)
                    mask[(row + 1) * width + column] = 255;
            }
        }
    }

    static void FastScalar(byte[] gray, byte[] mask, int width, int height, int low, int high, int minimum)
    {
        for (int row = 0; row < height; row++)
        {
            int column = 0;
            while (column < width)
            {
                int value = gray[row * width + column];
                if (value < low || value > high)
                {
                    mask[row * width + column] = 0;
                    column++;
                    continue;
                }
                int start = column;
                while (column < width && gray[row * width + column] >= low && gray[row * width + column] <= high)
                    column++;
                byte paint = column - start >= minimum ? (byte)255 : (byte)0;
                for (int at = start; at < column; at++)
                    mask[row * width + at] = paint;
            }
        }
    }
}
