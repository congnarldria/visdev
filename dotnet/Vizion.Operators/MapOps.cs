namespace Vizion;

public static partial class Operators
{
    /// <summary>先把灰階除以 255，做次方，再乘回 255。大於 1 會讓中間調變暗。彩色每個通道分開算。</summary>
    /// <remarks>灰階次方</remarks>
    /// <example>
    /// <code>Vizion.pow_image (Image, ImagePow, 2)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="exponent">次方，必須大於 0。</param>
    /// <returns>次方後的影像。</returns>
    public static ImageBuffer PowImage(ImageBuffer image, double exponent)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (!(exponent > 0))
            throw new ArgumentException("Exponent 必須大於 0");
        float power = (float)exponent;
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryGrayPow(source, destination, source.Length, exponent))
                return;
            for (int index = 0; index < source.Length; index++)
                destination[index] = RoundF((float)Math.Pow(source[index] / 255f, power) * 255f);
        });
    }

    /// <summary>每個通道分開開根號，再四捨五入成 byte。0 還是 0，81 變成 9，255 變成 16。</summary>
    /// <remarks>灰階開根號</remarks>
    /// <example>
    /// <code>Vizion.sqrt_image (Image, ImageSqrt)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <returns>開根號後的影像。</returns>
    public static ImageBuffer SqrtImage(ImageBuffer image)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryGraySqrt(source, destination, source.Length))
                return;
            for (int index = 0; index < source.Length; index++)
                destination[index] = RoundF((float)Math.Sqrt(source[index]));
        });
    }

    /// <summary>scaled 用 ln(灰階+1)／ln(256) 乘上 255，0 還是 0、255 還是 255。natural 是 round(ln(灰階))，0 維持 0，所以 255 只會變成 6。每個通道分開算。</summary>
    /// <remarks>灰階取對數</remarks>
    /// <example>
    /// <code>Vizion.log_image (Image, ImageLog, 'scaled')</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="mode">scaled 拉回 0 到 255，natural 是自然對數四捨五入。</param>
    /// <returns>對數後的影像。</returns>
    public static ImageBuffer LogImage(ImageBuffer image, string mode)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        bool natural = LogMode(mode);
        float scale = 255f / (float)Math.Log(256f);
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryGrayLog(source, destination, source.Length, natural))
                return;
            for (int index = 0; index < source.Length; index++)
            {
                if (!natural)
                    destination[index] = RoundF((float)Math.Log(source[index] + 1f) * scale);
                else if (source[index] < 1)
                    destination[index] = 0;
                else
                    destination[index] = RoundF((float)Math.Log(source[index]));
            }
        });
    }

    /// <summary>scaled 用 256 的次方再減 1，剛好是 scaled 對數的反運算：0 還是 0，128 約是 15，255 還是 255。natural 是 e 的灰階次方，超過 5 就會飽和。彩色每個通道各自做。</summary>
    /// <remarks>指數</remarks>
    /// <example>
    /// <code>Vizion.exp_image (Image, ImageExp, 'scaled')</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="mode">scaled 是對數的反運算，natural 是 e 的次方。</param>
    /// <returns>指數之後的影像。</returns>
    public static ImageBuffer ExpImage(ImageBuffer image, string mode)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        bool natural = LogMode(mode);
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryGrayExp(source, destination, source.Length, natural))
                return;
            for (int index = 0; index < source.Length; index++)
            {
                double mapped = natural
                    ? Math.Exp(source[index])
                    : Math.Pow(256.0, source[index] / 255.0) - 1.0;
                destination[index] = RoundD(mapped);
            }
        });
    }

    /// <summary>小於 Threshold 的灰階不變。其餘先對齊 Threshold 和 MaxGray，再做次方。Encode 為 true 時次方是 1/Gamma，會把中間調變亮；false 時次方就是 Gamma。彩色每個通道各自做。</summary>
    /// <remarks>伽瑪校正</remarks>
    /// <example>
    /// <code>Vizion.gamma_image (Image, ImageGamma, 2)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="gamma">大於 0。</param>
    /// <param name="offset">公式加上的灰階。</param>
    /// <param name="threshold">小於此值的灰階保持原樣。</param>
    /// <param name="maxGray">公式的最高灰階。</param>
    /// <param name="encode">true 用 1/Gamma，false 直接用 Gamma。</param>
    /// <returns>伽瑪校正後的影像。</returns>
    public static ImageBuffer GammaImage(ImageBuffer image, double gamma, double offset, double threshold, double maxGray, bool encode)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (!(gamma > 0))
            throw new ArgumentException("Gamma 必須大於 0");
        if (!(maxGray > threshold))
            throw new ArgumentException("MaxGray 必須大於 Threshold");
        float power = (float)gamma;
        float shift = (float)offset;
        float floor = (float)threshold;
        float ceiling = (float)maxGray;
        float exponent = encode ? 1f / power : power;
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryGrayGamma(source, destination, source.Length, gamma, offset, threshold, maxGray, encode))
                return;
            for (int index = 0; index < source.Length; index++)
            {
                float sample = source[index];
                if (sample < floor)
                {
                    destination[index] = RoundF(sample);
                    continue;
                }
                float ratio = (sample - floor) / (ceiling - floor);
                if (ratio < 0f)
                    ratio = 0f;
                else if (ratio > 1f)
                    ratio = 1f;
                destination[index] = RoundF(shift + (ceiling - shift) * (float)Math.Pow(ratio, exponent));
            }
        });
    }

    /// <summary>把 128 當作 0。平坦的 128 變成 0，0 變成 128，255 變成 127。彩色每個通道各自做。</summary>
    /// <remarks>對 128 取絕對值</remarks>
    /// <example>
    /// <code>Vizion.abs_image (Image, ImageAbs)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <returns>每個像素離 128 的距離。</returns>
    public static ImageBuffer AbsImage(ImageBuffer image)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryGrayAbs(source, destination, source.Length))
                return;
            for (int index = 0; index < source.Length; index++)
            {
                int signedValue = source[index] - 128;
                destination[index] = (byte)(signedValue < 0 ? -signedValue : signedValue);
            }
        });
    }

    /// <summary>小於 Min 變成 0，大於 Max 變成 255。Min 和 Max 相同時整張變成 0。彩色三個通道用同一組上下限。</summary>
    /// <remarks>把一段灰階拉開到 0 到 255</remarks>
    /// <example>
    /// <code>Vizion.scale_image_range (Image, ImageScaled, 0, 63)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="low">對應到 0 的灰階。</param>
    /// <param name="high">對應到 255 的灰階。</param>
    /// <returns>拉伸後的影像。定義域沿用輸入。</returns>
    public static ImageBuffer ScaleImageRange(ImageBuffer image, double low, double high)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        return Map(image, (source, destination) =>
        {
            if (NativeCore.TryGrayRange(source, destination, source.Length, low, high))
                return;
            if (Math.Abs(high - low) < 1e-9)
            {
                Array.Clear(destination, 0, source.Length);
                return;
            }
            float floor = (float)low;
            float span = (float)high - floor;
            for (int index = 0; index < source.Length; index++)
                destination[index] = Trunc(((source[index] - floor) / span) * 255f);
        });
    }

    /// <summary>每個像素、每個通道各自做 AND。8 位元，超過的位元不保留。定義域沿用第一張。</summary>
    /// <remarks>兩張影像做位元且</remarks>
    /// <example>
    /// <code>Vizion.bit_and (Image1, Image2, ImageAnd)</code>
    /// </example>
    /// <param name="left">第一張影像。</param>
    /// <param name="right">第二張影像。</param>
    /// <returns>位元且。定義域沿用第一張。</returns>
    public static ImageBuffer BitAnd(ImageBuffer left, ImageBuffer right) => Bits(left, right, 0);

    /// <summary>每個像素、每個通道各自做 OR。定義域沿用第一張。</summary>
    /// <remarks>兩張影像做位元或</remarks>
    /// <example>
    /// <code>Vizion.bit_or (Image1, Image2, ImageOr)</code>
    /// </example>
    /// <param name="left">第一張影像。</param>
    /// <param name="right">第二張影像。</param>
    /// <returns>位元或。定義域沿用第一張。</returns>
    public static ImageBuffer BitOr(ImageBuffer left, ImageBuffer right) => Bits(left, right, 1);

    /// <summary>每個像素、每個通道各自做 XOR。定義域沿用第一張。</summary>
    /// <remarks>兩張影像做位元互斥或</remarks>
    /// <example>
    /// <code>Vizion.bit_xor (Image1, Image2, ImageXor)</code>
    /// </example>
    /// <param name="left">第一張影像。</param>
    /// <param name="right">第二張影像。</param>
    /// <returns>位元互斥或。定義域沿用第一張。</returns>
    public static ImageBuffer BitXor(ImageBuffer left, ImageBuffer right) => Bits(left, right, 2);

    static ImageBuffer Bits(ImageBuffer left, ImageBuffer right, int op)
    {
        if (left == null || right == null)
            throw new ArgumentNullException(nameof(left));
        if (left.Width != right.Width || left.Height != right.Height || left.Channels != right.Channels)
            throw new ArgumentException("兩張影像的尺寸或通道數不同");
        var destination = new byte[left.Pixels.Length];
        if (!NativeCore.TryGrayBits(left.Pixels, right.Pixels, destination, destination.Length, op))
        {
            for (int index = 0; index < destination.Length; index++)
            {
                byte second = right.Pixels[index];
                destination[index] = op switch
                {
                    0 => (byte)(left.Pixels[index] & second),
                    1 => (byte)(left.Pixels[index] | second),
                    _ => (byte)(left.Pixels[index] ^ second),
                };
            }
        }
        return ImageBuffer.Wrap(left.Width, left.Height, left.Channels, destination, left.Domain);
    }

    static bool LogMode(string mode)
    {
        if (mode == null)
            throw new ArgumentNullException(nameof(mode));
        string name = mode.Trim().ToLowerInvariant();
        if (name == "scaled")
            return false;
        if (name == "natural")
            return true;
        throw new ArgumentException("Mode 只能是 scaled 或 natural");
    }

    static byte RoundF(float value)
    {
        if (!(value > 0f))
            return 0;
        if (value >= 255f)
            return 255;
        int whole = (int)value;
        float frac = value - whole;
        int rounded = frac > 0.5f ? whole + 1 : frac < 0.5f ? whole : ((whole & 1) != 0 ? whole + 1 : whole);
        return rounded > 255 ? (byte)255 : (byte)rounded;
    }

    static byte RoundD(double value)
    {
        if (!(value > 0.0))
            return 0;
        if (value >= 255.0)
            return 255;
        int whole = (int)value;
        double frac = value - whole;
        int rounded = frac > 0.5 ? whole + 1 : frac < 0.5 ? whole : ((whole & 1) != 0 ? whole + 1 : whole);
        return rounded > 255 ? (byte)255 : (byte)rounded;
    }
}
