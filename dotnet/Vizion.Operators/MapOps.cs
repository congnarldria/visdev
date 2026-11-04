namespace Vizion;

public static partial class Operators
{
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

    public static ImageBuffer BitAnd(ImageBuffer left, ImageBuffer right) => Bits(left, right, 0);

    public static ImageBuffer BitOr(ImageBuffer left, ImageBuffer right) => Bits(left, right, 1);

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
