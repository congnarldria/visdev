namespace Vizion;

public static partial class Operators
{
    /// <summary>8 位元的 NOT。0 變成 255，10 變成 245。彩色每個通道各自做。</summary>
    /// <remarks>影像做位元否定</remarks>
    /// <example>
    /// <code>Vizion.bit_not (Image, ImageNot)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <returns>8 位元否定。0 變成 255。</returns>
    public static ImageBuffer BitNot(ImageBuffer image)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryGrayBitConst(source, destination, source.Length, 0, 0))
                return;
            for (int index = 0; index < source.Length; index++)
                destination[index] = (byte)~source[index];
        });
    }

    /// <summary>每個像素和 BitMask 做位元 AND。遮罩 255 時灰階不變。</summary>
    /// <remarks>用固定遮罩做 AND</remarks>
    /// <example>
    /// <code>Vizion.bit_mask (Image, ImageMask, 240)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="mask">0 到 255 的遮罩。</param>
    /// <returns>和遮罩做位元且之後的影像。</returns>
    public static ImageBuffer BitMask(ImageBuffer image, int mask)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (mask < 0 || mask > 255)
            throw new ArgumentException("BitMask 要在 0 到 255");
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryGrayBitConst(source, destination, source.Length, 1, mask))
                return;
            for (int index = 0; index < source.Length; index++)
                destination[index] = (byte)(source[index] & mask);
        });
    }

    /// <summary>該位元是 1 的像素變成 255，否則是 0。Bit 要在 0 到 7。</summary>
    /// <remarks>抽出一個位元</remarks>
    /// <example>
    /// <code>Vizion.bit_slice (Image, ImageSlice, 0)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="bit">0 是最低位，要在 0 到 7。</param>
    /// <returns>該位元是 1 的像素為 255，否則是 0。</returns>
    public static ImageBuffer BitSlice(ImageBuffer image, int bit)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (bit < 0 || bit > 7)
            throw new ArgumentException("Bit 要在 0 到 7");
        int flag = 1 << bit;
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryGrayBitConst(source, destination, source.Length, 4, bit))
                return;
            for (int index = 0; index < source.Length; index++)
                destination[index] = (source[index] & flag) != 0 ? (byte)255 : (byte)0;
        });
    }

    /// <summary>每個通道乘上 2 的 Shift 次方，只留 8 個位元。1 左移 3 位是 8。移 8 位以上變成 0。</summary>
    /// <remarks>灰階左移</remarks>
    /// <example>
    /// <code>Vizion.bit_lshift (Image, ImageShift, 1)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="shift">左移的位數。</param>
    /// <returns>左移後的影像。移 8 位以上變成 0。</returns>
    public static ImageBuffer BitLShift(ImageBuffer image, int shift) => Shift(image, shift, left: true);

    /// <summary>每個通道除以 2 的 Shift 次方，小數直接丟掉。16 右移 2 位是 4。</summary>
    /// <remarks>灰階右移</remarks>
    /// <example>
    /// <code>Vizion.bit_rshift (Image, ImageShift, 1)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="shift">右移的位數。</param>
    /// <returns>右移後的影像。小數直接丟掉。</returns>
    public static ImageBuffer BitRShift(ImageBuffer image, int shift) => Shift(image, shift, left: false);

    /// <summary>128 代表和周圍一樣，比 128 大是比周圍亮。偶數遮罩會加 1，最大 31。邊界用鏡射。彩色先轉成亮度。</summary>
    /// <remarks>減去局部平均</remarks>
    /// <example>
    /// <code>Vizion.highpass_image (Image, Highpass, 5, 5)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="width">遮罩寬。</param>
    /// <param name="height">遮罩高。</param>
    /// <returns>高通結果。128 代表和周圍一樣。</returns>
    public static ImageBuffer HighpassImage(ImageBuffer image, int width, int height)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        int wide = OddSide(width, "Width", 31);
        int high = OddSide(height, "Height", 31);
        return GrayFilter(image, (source, destination) =>
        {
            if (NativeCore.TryHighpass(source, destination, image.Width, image.Height, wide, high))
                return;
            float[] plane = BytesToFloat(source);
            float[] mean = BoxFloat(plane, image.Width, image.Height, wide, high);
            for (int index = 0; index < source.Length; index++)
                destination[index] = RoundF(128f + source[index] - mean[index]);
        });
    }

    /// <summary>用矩形平均算出每個像素附近的標準差，再四捨五入成 0 到 255。彩色先轉成亮度。偶數遮罩可用，原點可能和 HALCON 差 1 像素。</summary>
    /// <remarks>局部標準差</remarks>
    /// <example>
    /// <code>Vizion.deviation_image (Image, ImageDeviation, 5, 5)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="width">遮罩寬。</param>
    /// <param name="height">遮罩高。</param>
    /// <returns>局部標準差。</returns>
    public static ImageBuffer DeviationImage(ImageBuffer image, int width, int height)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (width < 1 || height < 1)
            throw new ArgumentException("遮罩寬高至少要 1");
        return GrayFilter(image, (source, destination) =>
        {
            if (NativeCore.TryDeviation(source, destination, image.Width, image.Height, width, height))
                return;
            float[] plane = BytesToFloat(source);
            float[] mean = BoxFloat(plane, image.Width, image.Height, width, height);
            var squares = new float[plane.Length];
            for (int index = 0; index < plane.Length; index++)
                squares[index] = plane[index] * plane[index];
            float[] square = BoxFloat(squares, image.Width, image.Height, width, height);
            for (int index = 0; index < source.Length; index++)
            {
                float variance = square[index] - mean[index] * mean[index];
                if (variance < 0f)
                    variance = 0f;
                destination[index] = RoundF((float)Math.Sqrt(variance));
            }
        });
    }

    /// <summary>和 deviation_image 同一套矩形平均。寬、高順序相同，灰階也相同。彩色先轉成亮度。</summary>
    /// <remarks>局部標準差</remarks>
    /// <example>
    /// <code>Vizion.sigma_image (Image, ImageSigma, 5, 5)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="width">遮罩寬。</param>
    /// <param name="height">遮罩高。</param>
    /// <returns>局部標準差，和 deviation_image 相同。</returns>
    public static ImageBuffer SigmaImage(ImageBuffer image, int width, int height) =>
        DeviationImage(image, width, height);

    /// <summary>每個像素輸出附近最大值減最小值。彩色先轉成亮度。用來抓局部對比。</summary>
    /// <remarks>矩形範圍的灰階差</remarks>
    /// <example>
    /// <code>Vizion.gray_range_rect (Image, ImageResult, 5, 5)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="maskHeight">遮罩高。</param>
    /// <param name="maskWidth">遮罩寬。</param>
    /// <returns>局部最大減最小。</returns>
    public static ImageBuffer GrayRangeRect(ImageBuffer image, int maskHeight, int maskWidth)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (maskHeight < 1 || maskWidth < 1)
            throw new ArgumentException("遮罩寬高至少要 1");
        return GrayFilter(image, (source, destination) =>
        {
            if (NativeCore.TryGraySpan(source, destination, image.Width, image.Height, maskHeight, maskWidth))
                return;
            var gray = ImageBuffer.Gray(image.Width, image.Height, source);
            byte[] bright = GrayDilation(gray, maskHeight, maskWidth).Pixels;
            byte[] dark = GrayErosion(gray, maskHeight, maskWidth).Pixels;
            for (int index = 0; index < destination.Length; index++)
                destination[index] = (byte)(bright[index] - dark[index]);
        });
    }

    /// <summary>3×3 的 n_4 是上下左右，n_8 再包含對角。5 和 7 用 OpenCV 的拉普拉斯，兩種 FilterMask 相同。signed 會加上 128 再夾到 0 到 255。</summary>
    /// <remarks>拉普拉斯</remarks>
    /// <example>
    /// <code>Vizion.laplace (Image, ImageLaplace, 'absolute', 3, 'n_4')</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="resultType">absolute 取絕對值，signed 以 128 為 0。</param>
    /// <param name="maskSize">3、5 或 7。</param>
    /// <param name="filterMask">n_4 或 n_8。只在 3×3 有差別。</param>
    /// <returns>拉普拉斯結果。</returns>
    public static ImageBuffer Laplace(ImageBuffer image, string resultType, int maskSize, string filterMask)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        bool absolute = LaplaceAbsolute(resultType);
        if (maskSize != 3 && maskSize != 5 && maskSize != 7)
            throw new ArgumentException("MaskSize 只能是 3、5 或 7");
        bool eight = LaplaceEight(filterMask);
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryLaplace(source, destination, image.Width, image.Height, maskSize, eight, absolute))
                return;
            LaplaceScalar(source, destination, image.Width, image.Height, maskSize, eight, absolute);
        });
    }

    static ImageBuffer Shift(ImageBuffer image, int shift, bool left)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (shift < 0)
            throw new ArgumentException("Shift 不能是負的");
        int op = left ? 2 : 3;
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryGrayBitConst(source, destination, source.Length, op, shift))
                return;
            for (int index = 0; index < source.Length; index++)
            {
                if (shift >= 8)
                    destination[index] = 0;
                else if (left)
                    destination[index] = (byte)(source[index] << shift);
                else
                    destination[index] = (byte)(source[index] >> shift);
            }
        });
    }

    static ImageBuffer GrayFilter(ImageBuffer image, Action<byte[], byte[]> filter)
    {
        byte[] source = image.GrayBytes();
        var destination = new byte[source.Length];
        filter(source, destination);
        return ImageBuffer.Wrap(image.Width, image.Height, 1, destination, image.Domain);
    }

    static float[] BytesToFloat(byte[] source)
    {
        var plane = new float[source.Length];
        for (int index = 0; index < source.Length; index++)
            plane[index] = source[index];
        return plane;
    }

    static int OddSide(int value, string name, int limit)
    {
        int side = value | 1;
        if (side < 1 || side > limit)
            throw new ArgumentException(name + " 要在 1 到 " + limit);
        return side;
    }

    static bool LaplaceAbsolute(string resultType)
    {
        string name = (resultType ?? "").Trim().ToLowerInvariant();
        if (name == "absolute")
            return true;
        if (name == "signed")
            return false;
        throw new ArgumentException("ResultType 只能是 absolute 或 signed");
    }

    static bool LaplaceEight(string filterMask)
    {
        string name = (filterMask ?? "").Trim().ToLowerInvariant();
        if (name == "n_4")
            return false;
        if (name == "n_8")
            return true;
        throw new ArgumentException("FilterMask 只能是 n_4 或 n_8");
    }

    static void LaplaceScalar(byte[] source, byte[] destination, int width, int height, int ksize, bool eight, bool absolute)
    {
        var plane = BytesToFloat(source);
        float[] response = ksize == 3
            ? Conv3(plane, width, height, eight)
            : LaplaceWide(plane, width, height, ksize);
        for (int index = 0; index < destination.Length; index++)
        {
            float value = absolute ? Math.Abs(response[index]) : response[index] + 128f;
            destination[index] = Trunc(value);
        }
    }

    static float[] Conv3(float[] source, int width, int height, bool eight)
    {
        float[] kernel = eight
            ? new float[] { 1, 1, 1, 1, -8, 1, 1, 1, 1 }
            : new float[] { 0, 1, 0, 1, -4, 1, 0, 1, 0 };
        var destination = new float[source.Length];
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                float sum = 0f;
                for (int dy = 0; dy < 3; dy++)
                {
                    int yy = FilterReflect(row + dy - 1, height);
                    for (int dx = 0; dx < 3; dx++)
                    {
                        int xx = FilterReflect(column + dx - 1, width);
                        sum += source[yy * width + xx] * kernel[dy * 3 + dx];
                    }
                }
                destination[row * width + column] = sum;
            }
        }
        return destination;
    }

    static float[] LaplaceWide(float[] source, int width, int height, int ksize)
    {
        float[] second = ksize == 5
            ? new float[] { 1, 0, -2, 0, 1 }
            : new float[] { 1, 2, -1, -4, -1, 2, 1 };
        float[] smooth = ksize == 5
            ? new float[] { 1, 4, 6, 4, 1 }
            : new float[] { 1, 6, 15, 20, 15, 6, 1 };
        float[] across = SepFilter(source, width, height, second, smooth);
        float[] along = SepFilter(source, width, height, smooth, second);
        for (int index = 0; index < across.Length; index++)
            across[index] += along[index];
        return across;
    }

    static float[] SepFilter(float[] source, int width, int height, float[] across, float[] along)
    {
        int anchorX = across.Length / 2;
        int anchorY = along.Length / 2;
        var horizontal = new float[source.Length];
        var destination = new float[source.Length];
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                float sum = 0f;
                for (int tap = 0; tap < across.Length; tap++)
                    sum += source[row * width + FilterReflect(column + tap - anchorX, width)] * across[tap];
                horizontal[row * width + column] = sum;
            }
        }
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                float sum = 0f;
                for (int tap = 0; tap < along.Length; tap++)
                    sum += horizontal[FilterReflect(row + tap - anchorY, height) * width + column] * along[tap];
                destination[row * width + column] = sum;
            }
        }
        return destination;
    }
}
