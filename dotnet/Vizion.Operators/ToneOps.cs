namespace Vizion;

public static partial class Operators
{
    /// <summary>輸出 = 原圖 + (原圖 − 局部平均) × Factor。</summary>
    /// <remarks>強化邊緣對比</remarks>
    /// <example>
    /// <code>Vizion.emphasize (Image, Sharp, 7, 7, 1)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="maskWidth">遮罩寬。</param>
    /// <param name="maskHeight">遮罩高。</param>
    /// <param name="factor">強化係數。</param>
    /// <returns>強化後的影像。</returns>
    public static ImageBuffer Emphasize(ImageBuffer image, int maskWidth, int maskHeight, double factor)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        ImageBuffer blur = MeanImage(image, OddAtLeastOne(maskWidth), OddAtLeastOne(maskHeight));
        var destination = new byte[image.Pixels.Length];
        Apply(image.Pixels, blur.Pixels, destination, 9, factor, 0);
        return ImageBuffer.Wrap(image.Width, image.Height, image.Channels, destination, image.Domain);
    }

    /// <summary>灰階直接等化。彩色影像只等化亮度通道。</summary>
    /// <remarks>直方圖等化</remarks>
    /// <example>
    /// <code>Vizion.equ_histo_image (Image, Equ)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <returns>直方圖等化結果。彩色只等化亮度。</returns>
    public static ImageBuffer EquHistoImage(ImageBuffer image)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        var destination = new byte[image.Pixels.Length];
        if (image.Channels == 1)
        {
            if (!NativeCore.TryEqualize(image.Pixels, destination, image.Pixels.Length))
                Equalize(image.Pixels, destination, image.Pixels.Length);
        }
        else if (!NativeCore.TryEqualizeBgr(image.Pixels, destination, image.Width * image.Height))
        {
            EqualizeBgr(image.Pixels, destination, image.Width * image.Height);
        }
        return ImageBuffer.Wrap(image.Width, image.Height, image.Channels, destination, image.Domain);
    }

    /// <summary>定義域裡的最小值變成 0、最大值變成 255。整張一樣亮時輸出 0。彩色圖每個通道分開拉。</summary>
    /// <remarks>把對比拉滿</remarks>
    /// <example>
    /// <code>Vizion.scale_image_max (Image, ImageScaleMax)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <returns>對比拉到 0 到 255 的影像。定義域外是 0。</returns>
    public static ImageBuffer ScaleImageMax(ImageBuffer image)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        byte[]? domain = ByteDomain(image);
        if (image.Channels == 1)
        {
            var destination = new byte[image.Pixels.Length];
            if (!NativeCore.TryScaleMax(image.Pixels, destination, image.Pixels.Length, domain))
                ScaleMax(image.Pixels, destination, image.Pixels.Length, domain);
            return ImageBuffer.Wrap(image.Width, image.Height, 1, destination, image.Domain);
        }
        int plane = image.Width * image.Height;
        var merged = new byte[plane * 3];
        var source = new byte[plane];
        var stretched = new byte[plane];
        for (int channel = 0; channel < 3; channel++)
        {
            CopyPlane(image.Pixels, source, plane, channel);
            if (!NativeCore.TryScaleMax(source, stretched, plane, domain))
                ScaleMax(source, stretched, plane, domain);
            StorePlane(merged, stretched, plane, channel);
        }
        return ImageBuffer.Wrap(image.Width, image.Height, 3, merged, image.Domain);
    }

    /// <summary>輸出 = (原圖 − 局部平均) + 局部平均 × Factor + 整張平均 × (1 − Factor)。Factor 1 就是原圖，Factor 0 去掉局部明暗。偶數遮罩會加 1 變成奇數。這和 emphasize 不同，emphasize 是把高頻加回原圖。</summary>
    /// <remarks>拉平緩慢的照明變化</remarks>
    /// <example>
    /// <code>Vizion.illuminate (Image, ImageIlluminate, 31, 31, 0)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="maskWidth">局部平均的寬。</param>
    /// <param name="maskHeight">局部平均的高。</param>
    /// <param name="factor">0 完全拉平，1 保持原圖。</param>
    /// <returns>校正照明後的影像。</returns>
    public static ImageBuffer Illuminate(ImageBuffer image, int maskWidth, int maskHeight, double factor)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        int width = OddAtLeast(maskWidth, "MaskWidth");
        int height = OddAtLeast(maskHeight, "MaskHeight");
        byte[]? domain = ByteDomain(image);
        if (image.Channels == 1)
        {
            var destination = new byte[image.Pixels.Length];
            if (!NativeCore.TryIlluminate(image.Pixels, destination, image.Width, image.Height, width, height, factor, domain))
                IlluminatePlane(image.Pixels, destination, image.Width, image.Height, width, height, factor, domain);
            return ImageBuffer.Wrap(image.Width, image.Height, 1, destination, image.Domain);
        }
        int plane = image.Width * image.Height;
        var merged = new byte[plane * 3];
        var source = new byte[plane];
        var corrected = new byte[plane];
        for (int channel = 0; channel < 3; channel++)
        {
            CopyPlane(image.Pixels, source, plane, channel);
            if (!NativeCore.TryIlluminate(source, corrected, image.Width, image.Height, width, height, factor, domain))
                IlluminatePlane(source, corrected, image.Width, image.Height, width, height, factor, domain);
            StorePlane(merged, corrected, plane, channel);
        }
        return ImageBuffer.Wrap(image.Width, image.Height, 3, merged, image.Domain);
    }

    /// <summary>灰階是 1。彩色是 3，通道 1 紅、2 綠、3 藍。</summary>
    /// <remarks>通道數</remarks>
    /// <example>
    /// <code>Vizion.count_channels (Image, Channels)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <returns>灰階是 1，彩色是 3。</returns>
    public static int CountChannels(ImageBuffer image)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        return image.Channels;
    }

    /// <summary>彩色圖的通道 1 是紅、2 是綠、3 是藍。影像在內部以 BGR 存放，這個編號仍跟 HALCON 相同。</summary>
    /// <remarks>取出一個通道</remarks>
    /// <example>
    /// <code>Vizion.access_channel (Image, Red, 1)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="channel">通道編號，從 1 開始。1 紅、2 綠、3 藍。</param>
    /// <returns>單通道影像。</returns>
    public static ImageBuffer AccessChannel(ImageBuffer image, int channel)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (image.Channels == 1)
        {
            if (channel != 1)
                throw new ArgumentException("灰階影像只有通道 1");
            var copy = new byte[image.Pixels.Length];
            Buffer.BlockCopy(image.Pixels, 0, copy, 0, copy.Length);
            return ImageBuffer.Wrap(image.Width, image.Height, 1, copy, image.Domain);
        }
        int memory = ChannelIndex(channel);
        int plane = image.Width * image.Height;
        var destination = new byte[plane];
        CopyPlane(image.Pixels, destination, plane, memory);
        return ImageBuffer.Wrap(image.Width, image.Height, 1, destination, image.Domain);
    }

    /// <summary>三張都必須是相同尺寸的灰階圖。順序是紅、綠、藍。</summary>
    /// <remarks>三張灰階合成彩色</remarks>
    /// <example>
    /// <code>Vizion.compose3 (Red, Green, Blue, ImageRGB)</code>
    /// </example>
    /// <param name="red">紅通道。</param>
    /// <param name="green">綠通道。</param>
    /// <param name="blue">藍通道。</param>
    /// <returns>彩色影像。定義域是三張圖的交集。</returns>
    public static ImageBuffer Compose3(ImageBuffer red, ImageBuffer green, ImageBuffer blue)
    {
        RequireGray(red, green, blue, "compose3 的三張輸入都要是灰階");
        int plane = red.Width * red.Height;
        var merged = new byte[plane * 3];
        if (!NativeCore.TryMergeBgr(blue.Pixels, green.Pixels, red.Pixels, merged, plane))
        {
            for (int index = 0; index < plane; index++)
            {
                merged[index * 3] = blue.Pixels[index];
                merged[index * 3 + 1] = green.Pixels[index];
                merged[index * 3 + 2] = red.Pixels[index];
            }
        }
        return ImageBuffer.Wrap(red.Width, red.Height, 3, merged, FoldDomain(red, green, blue));
    }

    /// <summary>灰階影像不能拆。通道順序是紅、綠、藍。</summary>
    /// <remarks>彩色拆成紅綠藍</remarks>
    /// <example>
    /// <code>Vizion.decompose3 (ImageRGB, Red, Green, Blue)</code>
    /// </example>
    /// <param name="image">彩色影像。</param>
    /// <returns>依序是紅、綠、藍。不取代目前影像。</returns>
    public static ImageBuffer[] Decompose3(ImageBuffer image)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (image.Channels != 3)
            throw new ArgumentException("decompose3 需要三通道彩色影像");
        int plane = image.Width * image.Height;
        var blue = new byte[plane];
        var green = new byte[plane];
        var red = new byte[plane];
        if (!NativeCore.TrySplitBgr(image.Pixels, blue, green, red, plane))
        {
            CopyPlane(image.Pixels, blue, plane, 0);
            CopyPlane(image.Pixels, green, plane, 1);
            CopyPlane(image.Pixels, red, plane, 2);
        }
        return new[]
        {
            ImageBuffer.Wrap(image.Width, image.Height, 1, red, image.Domain),
            ImageBuffer.Wrap(image.Width, image.Height, 1, green, image.Domain),
            ImageBuffer.Wrap(image.Width, image.Height, 1, blue, image.Domain),
        };
    }

    /// <summary>三張輸入都要是相同尺寸的灰階。hsv 的 H 是 0 到 180。lab 的 a、b 以 128 為 0。這是 OpenCV 的數值，不是 HALCON 的實數範圍。</summary>
    /// <remarks>把紅綠藍轉成其他色彩空間</remarks>
    /// <example>
    /// <code>Vizion.trans_from_rgb (Red, Green, Blue, H, S, V, 'hsv')</code>
    /// </example>
    /// <param name="red">紅。</param>
    /// <param name="green">綠。</param>
    /// <param name="blue">藍。</param>
    /// <param name="colorSpace">hsv、lab 或 yuv。</param>
    /// <returns>依序是轉換後的三個通道。不取代目前影像。</returns>
    public static ImageBuffer[] TransFromRgb(ImageBuffer red, ImageBuffer green, ImageBuffer blue, string colorSpace)
    {
        RequireGray(red, green, blue, "三張輸入都要是相同尺寸的灰階");
        int space = ColorSpace(colorSpace);
        int plane = red.Width * red.Height;
        var first = new byte[plane];
        var second = new byte[plane];
        var third = new byte[plane];
        if (!NativeCore.TryRgbConvert(red.Pixels, green.Pixels, blue.Pixels, first, second, third, plane, space))
            ConvertRgb(red.Pixels, green.Pixels, blue.Pixels, first, second, third, plane, space);
        Region? domain = red.Domain;
        return new[]
        {
            ImageBuffer.Wrap(red.Width, red.Height, 1, first, domain),
            ImageBuffer.Wrap(red.Width, red.Height, 1, second, domain),
            ImageBuffer.Wrap(red.Width, red.Height, 1, third, domain),
        };
    }

    static int OddAtLeastOne(int value)
    {
        int odd = value | 1;
        return odd < 1 ? 1 : odd;
    }

    static int OddAtLeast(int value, string name)
    {
        if (value < 1)
            throw new ArgumentOutOfRangeException(name, name + " 至少要 1");
        return value | 1;
    }

    static int ChannelIndex(int channel)
    {
        if (channel == 1)
            return 2;
        if (channel == 2)
            return 1;
        if (channel == 3)
            return 0;
        throw new ArgumentException("通道 " + channel + " 不存在");
    }

    static int ColorSpace(string colorSpace)
    {
        if (colorSpace == null)
            throw new ArgumentNullException(nameof(colorSpace));
        string name = colorSpace.Trim().ToLowerInvariant();
        if (name == "hsv")
            return 0;
        if (name == "lab")
            return 1;
        if (name == "yuv")
            return 2;
        throw new ArgumentException("ColorSpace 只能是 hsv、lab 或 yuv");
    }

    static void RequireGray(ImageBuffer red, ImageBuffer green, ImageBuffer blue, string message)
    {
        if (red == null || green == null || blue == null)
            throw new ArgumentNullException(nameof(red));
        if (red.Channels != 1 || green.Channels != 1 || blue.Channels != 1)
            throw new ArgumentException(message);
        if (green.Width != red.Width || green.Height != red.Height || blue.Width != red.Width || blue.Height != red.Height)
            throw new ArgumentException("三張灰階的尺寸要相同");
    }

    static Region? FoldDomain(ImageBuffer red, ImageBuffer green, ImageBuffer blue)
    {
        Region? domain = red.Domain;
        domain = AndOptional(domain, green.Domain);
        return AndOptional(domain, blue.Domain);
    }

    static Region? AndOptional(Region? left, Region? right)
    {
        if (left == null)
            return right;
        if (right == null)
            return left;
        return Intersection(left, right);
    }

    static void CopyPlane(byte[] source, byte[] plane, int count, int channel)
    {
        for (int index = 0; index < count; index++)
            plane[index] = source[index * 3 + channel];
    }

    static void StorePlane(byte[] merged, byte[] plane, int count, int channel)
    {
        for (int index = 0; index < count; index++)
            merged[index * 3 + channel] = plane[index];
    }

    static byte[]? ByteDomain(ImageBuffer image)
    {
        Region? domain = image.Domain;
        if (domain == null)
            return null;
        var mask = new byte[image.Width * image.Height];
        for (int index = 0; index < domain.Rows.Length; index++)
        {
            int row = domain.Rows[index];
            int column = domain.Columns[index];
            int length = domain.Lengths[index];
            int offset = row * image.Width + column;
            for (int step = 0; step < length; step++)
                mask[offset + step] = 255;
        }
        return mask;
    }

    static void Equalize(byte[] source, byte[] destination, int count)
    {
        var hist = new int[256];
        for (int index = 0; index < count; index++)
            hist[source[index]]++;
        int first = 0;
        while (first < 256 && hist[first] == 0)
            first++;
        if (first == 256 || hist[first] == count)
        {
            Buffer.BlockCopy(source, 0, destination, 0, count);
            return;
        }
        double scale = 255.0 / (count - hist[first]);
        var lut = new byte[256];
        int sum = 0;
        for (int index = first + 1; index < 256; index++)
        {
            sum += hist[index];
            lut[index] = (byte)RoundU8(sum * scale);
        }
        for (int index = 0; index < count; index++)
            destination[index] = lut[source[index]];
    }

    static void EqualizeBgr(byte[] source, byte[] destination, int count)
    {
        var y = new byte[count];
        var cr = new byte[count];
        var cb = new byte[count];
        for (int index = 0; index < count; index++)
        {
            int yy, rr, bb;
            BgrToYCrCb(source[index * 3], source[index * 3 + 1], source[index * 3 + 2], out yy, out rr, out bb);
            y[index] = (byte)yy;
            cr[index] = (byte)rr;
            cb[index] = (byte)bb;
        }
        Equalize(y, y, count);
        for (int index = 0; index < count; index++)
            YCrCbToBgr(y[index], cr[index], cb[index], destination, index * 3);
    }

    static void ScaleMax(byte[] source, byte[] destination, int count, byte[]? domain)
    {
        int lo = 255;
        int hi = 0;
        bool any = false;
        for (int index = 0; index < count; index++)
        {
            if (domain != null && domain[index] == 0)
                continue;
            any = true;
            if (source[index] < lo)
                lo = source[index];
            if (source[index] > hi)
                hi = source[index];
        }
        if (!any || hi <= lo)
        {
            Array.Clear(destination, 0, count);
            return;
        }
        double scale = 255.0 / (hi - lo);
        for (int index = 0; index < count; index++)
        {
            if (domain != null && domain[index] == 0)
            {
                destination[index] = 0;
                continue;
            }
            destination[index] = TruncDouble(((double)source[index] - lo) * scale);
        }
    }

    static void IlluminatePlane(byte[] source, byte[] destination, int width, int height, int maskWidth, int maskHeight, double factor, byte[]? domain)
    {
        int count = width * height;
        float sum = 0f;
        int valid = 0;
        for (int index = 0; index < count; index++)
        {
            if (domain != null && domain[index] == 0)
                continue;
            sum += source[index];
            valid++;
        }
        float fill = valid == 0 ? 0f : sum / valid;
        var work = new float[count];
        for (int index = 0; index < count; index++)
            work[index] = domain != null && domain[index] == 0 ? fill : source[index];
        float[] blur = BoxFloat(work, width, height, maskWidth, maskHeight);
        for (int index = 0; index < count; index++)
        {
            if (domain != null && domain[index] == 0)
            {
                destination[index] = 0;
                continue;
            }
            float diff = source[index] - blur[index];
            float value = diff + blur[index] * (float)factor + fill * (float)(1.0 - factor);
            destination[index] = Trunc(value);
        }
    }

    static float[] BoxFloat(float[] source, int width, int height, int maskWidth, int maskHeight)
    {
        int anchorX = maskWidth / 2;
        int anchorY = maskHeight / 2;
        double scale = 1.0 / (maskWidth * (double)maskHeight);
        var horizontal = new double[width * height];
        for (int row = 0; row < height; row++)
        {
            int offset = row * width;
            double sum = 0;
            for (int tap = 0; tap < maskWidth; tap++)
                sum += source[offset + FilterReflect(tap - anchorX, width)];
            horizontal[offset] = sum;
            for (int column = 1; column < width; column++)
            {
                sum += source[offset + FilterReflect(column + maskWidth - 1 - anchorX, width)];
                sum -= source[offset + FilterReflect(column - 1 - anchorX, width)];
                horizontal[offset + column] = sum;
            }
        }
        var destination = new float[width * height];
        for (int column = 0; column < width; column++)
        {
            double sum = 0;
            for (int tap = 0; tap < maskHeight; tap++)
                sum += horizontal[FilterReflect(tap - anchorY, height) * width + column];
            destination[column] = (float)(sum * scale);
            for (int row = 1; row < height; row++)
            {
                sum += horizontal[FilterReflect(row + maskHeight - 1 - anchorY, height) * width + column];
                sum -= horizontal[FilterReflect(row - 1 - anchorY, height) * width + column];
                destination[row * width + column] = (float)(sum * scale);
            }
        }
        return destination;
    }

    static void ConvertRgb(byte[] red, byte[] green, byte[] blue, byte[] first, byte[] second, byte[] third, int count, int space)
    {
        for (int index = 0; index < count; index++)
        {
            if (space == 0)
                RgbToHsv(red[index], green[index], blue[index], first, second, third, index);
            else if (space == 1)
                RgbToLab(red[index], green[index], blue[index], first, second, third, index);
            else
                RgbToYuv(red[index], green[index], blue[index], first, second, third, index);
        }
    }

    static int[]? HsvSdiv;
    static int[]? HsvHdiv;

    static void EnsureHsv()
    {
        if (HsvSdiv != null)
            return;
        var sdiv = new int[256];
        var hdiv = new int[256];
        for (int index = 1; index < 256; index++)
        {
            sdiv[index] = RoundPos((255 << 12) / (double)index);
            hdiv[index] = RoundPos((180 << 12) / (6.0 * index));
        }
        HsvSdiv = sdiv;
        HsvHdiv = hdiv;
    }

    static void RgbToHsv(byte r, byte g, byte b, byte[] h, byte[] s, byte[] v, int index)
    {
        EnsureHsv();
        int vmax = r;
        int vmin = r;
        if (g > vmax)
            vmax = g;
        if (b > vmax)
            vmax = b;
        if (g < vmin)
            vmin = g;
        if (b < vmin)
            vmin = b;
        int diff = vmax - vmin;
        int vr = vmax == r ? -1 : 0;
        int vg = vmax == g ? -1 : 0;
        int sat = (diff * HsvSdiv![vmax] + (1 << 11)) >> 12;
        int hue = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
        hue = (hue * HsvHdiv![diff] + (1 << 11)) >> 12;
        if (hue < 0)
            hue += 180;
        h[index] = (byte)hue;
        s[index] = (byte)sat;
        v[index] = (byte)vmax;
    }

    static readonly int[] LabCoeff = { 1777, 1541, 778, 871, 2929, 296, 73, 448, 3575 };

    static void RgbToLab(byte r, byte g, byte b, byte[] l, byte[] a, byte[] bb, int index)
    {
        int red = LabTabs.Gamma[r];
        int green = LabTabs.Gamma[g];
        int blue = LabTabs.Gamma[b];
        int ix = Descale(red * LabCoeff[0] + green * LabCoeff[1] + blue * LabCoeff[2], 12);
        int iy = Descale(red * LabCoeff[3] + green * LabCoeff[4] + blue * LabCoeff[5], 12);
        int iz = Descale(red * LabCoeff[6] + green * LabCoeff[7] + blue * LabCoeff[8], 12);
        int fx = LabTabs.Cbrt[ix];
        int fy = LabTabs.Cbrt[iy];
        int fz = LabTabs.Cbrt[iz];
        l[index] = (byte)Sat(Descale(296 * fy - 1336934, 15));
        a[index] = (byte)Sat(Descale(500 * (fx - fy) + 128 * (1 << 15), 15));
        bb[index] = (byte)Sat(Descale(200 * (fy - fz) + 128 * (1 << 15), 15));
    }

    static void RgbToYuv(byte r, byte g, byte b, byte[] y, byte[] u, byte[] v, int index)
    {
        int yy = Descale(r * 4899 + g * 9617 + b * 1868, 14);
        int uu = Descale((b - yy) * 8061 + (128 << 14), 14);
        int vv = Descale((r - yy) * 14369 + (128 << 14), 14);
        y[index] = (byte)Sat(yy);
        u[index] = (byte)Sat(uu);
        v[index] = (byte)Sat(vv);
    }

    static void BgrToYCrCb(byte b, byte g, byte r, out int y, out int cr, out int cb)
    {
        int yy = Descale(b * 1868 + g * 9617 + r * 4899, 14);
        y = yy;
        cr = Sat(Descale((r - yy) * 11682 + (128 << 14), 14));
        cb = Sat(Descale((b - yy) * 9241 + (128 << 14), 14));
    }

    static void YCrCbToBgr(int y, int cr, int cb, byte[] destination, int offset)
    {
        destination[offset] = (byte)Sat(y + Descale((cb - 128) * 29049, 14));
        destination[offset + 1] = (byte)Sat(y + Descale((cb - 128) * -5636 + (cr - 128) * -11698, 14));
        destination[offset + 2] = (byte)Sat(y + Descale((cr - 128) * 22987, 14));
    }

    static int Descale(int value, int shift) => (value + (1 << (shift - 1))) >> shift;

    static int Sat(int value)
    {
        if (value < 0)
            return 0;
        if (value > 255)
            return 255;
        return value;
    }

    static int RoundPos(double value)
    {
        int whole = (int)value;
        double frac = value - whole;
        if (frac > 0.5)
            return whole + 1;
        if (frac < 0.5)
            return whole;
        return (whole & 1) != 0 ? whole + 1 : whole;
    }

    static byte TruncDouble(double value)
    {
        if (value <= 0)
            return 0;
        if (value >= 255)
            return 255;
        return (byte)value;
    }

    static int RoundU8(double value)
    {
        if (value <= 0)
            return 0;
        if (value >= 255)
            return 255;
        int rounded = RoundPos(value);
        return rounded > 255 ? 255 : rounded;
    }
}
