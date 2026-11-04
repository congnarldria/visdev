namespace Vizion;

public static partial class Operators
{
    /// <summary>整張都是同一個複數。可用來和頻譜相乘。</summary>
    /// <remarks>建立常數複數影像。</remarks>
    /// <example>
    /// <code>Vizion.gen_complex_const (1, 0, 80, 60, Complex)</code>
    /// </example>
    /// <param name="real">實部。</param>
    /// <param name="imag">虛部。</param>
    /// <param name="width">寬。</param>
    /// <param name="height">高。</param>
    /// <returns>常數複數影像。</returns>
    public static ComplexBuffer GenComplexConst(double real, double imag, int width, int height)
    {
        if (width < 1 || height < 1)
            throw new ArgumentException("寬和高至少要 1");
        int count = checked(width * height);
        var re = new double[count];
        var im = new double[count];
        if (!NativeCore.TryComplexFill(re, im, real, imag))
            ComplexFill(re, im, real, imag);
        return ComplexBuffer.Wrap(width, height, re, im);
    }

    /// <summary>一次讀一個點。原點在左上角。常數影像的能量全在 (0, 0)，數值是灰階乘上寬乘高。</summary>
    /// <remarks>讀一個頻譜點。</remarks>
    /// <example>
    /// <code>Vizion.get_complex (Complex, 0, 0, Real, Imag)</code>
    /// </example>
    /// <param name="spectrum">複數頻譜。</param>
    /// <param name="row">列。</param>
    /// <param name="column">行。</param>
    /// <param name="real">實部。</param>
    /// <param name="imag">虛部。</param>
    public static void GetComplex(ComplexBuffer spectrum, double row, double column, out double real, out double imag)
    {
        if (spectrum == null)
            throw new ArgumentNullException(nameof(spectrum));
        int itemRow = (int)Math.Round(row);
        int itemColumn = (int)Math.Round(column);
        if (itemRow < 0 || itemColumn < 0 || itemRow >= spectrum.Height || itemColumn >= spectrum.Width)
            throw new ArgumentException("座標超出複數影像");
        int index = itemRow * spectrum.Width + itemColumn;
        real = spectrum.Real[index];
        imag = spectrum.Imag[index];
    }

    /// <summary>center 把左上角的低頻搬到中央。corner 做相反的搬移。寬和高都是偶數時，兩種搬法一樣。</summary>
    /// <remarks>移動頻譜原點。</remarks>
    /// <example>
    /// <code>Vizion.fftshift (Complex, 'center', ComplexShifted)</code>
    /// </example>
    /// <param name="spectrum">複數頻譜。</param>
    /// <param name="mode">center 或 corner。</param>
    /// <returns>移動後的頻譜。</returns>
    public static ComplexBuffer FftShift(ComplexBuffer spectrum, string mode)
    {
        if (spectrum == null)
            throw new ArgumentNullException(nameof(spectrum));
        int center = ShiftCode(mode);
        var real = new double[spectrum.Real.Length];
        var imag = new double[spectrum.Imag.Length];
        if (!NativeCore.TryComplexShift(spectrum.Real, spectrum.Imag, real, imag, spectrum.Width, spectrum.Height, center))
            ComplexShift(spectrum.Real, spectrum.Imag, real, imag, spectrum.Width, spectrum.Height, center);
        return ComplexBuffer.Wrap(spectrum.Width, spectrum.Height, real, imag);
    }

    /// <summary>每個頻率都乘上同一個數。反轉換後，灰階也變成原來的倍率。</summary>
    /// <remarks>縮放頻譜。</remarks>
    /// <example>
    /// <code>Vizion.scale_fft (Complex, 0.5, ComplexScaled)</code>
    /// </example>
    /// <param name="spectrum">複數頻譜。</param>
    /// <param name="factor">倍率。</param>
    /// <returns>縮放後的頻譜。</returns>
    public static ComplexBuffer ScaleFft(ComplexBuffer spectrum, double factor)
    {
        return ComplexUnary(spectrum, (real, imag, dstReal, dstImag) =>
        {
            if (!NativeCore.TryComplexScale(real, imag, dstReal, dstImag, factor))
                ComplexScale(real, imag, dstReal, dstImag, factor);
        });
    }

    /// <summary>虛部變號，實部不變。</summary>
    /// <remarks>取頻譜的共軛。</remarks>
    /// <example>
    /// <code>Vizion.conj_fft (Complex, ComplexConj)</code>
    /// </example>
    /// <param name="spectrum">複數頻譜。</param>
    /// <returns>共軛頻譜。</returns>
    public static ComplexBuffer ConjFft(ComplexBuffer spectrum)
    {
        return ComplexUnary(spectrum, (real, imag, dstReal, dstImag) =>
        {
            if (!NativeCore.TryComplexConj(real, imag, dstReal, dstImag))
                ComplexConj(real, imag, dstReal, dstImag);
        });
    }

    /// <summary>尺寸要相同。常數影像的頻譜加自己，反轉換後灰階變成兩倍。</summary>
    /// <remarks>頻譜相加。</remarks>
    /// <example>
    /// <code>Vizion.add_fft (Complex1, Complex2, ComplexSum)</code>
    /// </example>
    /// <param name="left">第一張複數影像。</param>
    /// <param name="right">第二張複數影像。</param>
    /// <returns>逐點相加。</returns>
    public static ComplexBuffer AddFft(ComplexBuffer left, ComplexBuffer right) =>
        ComplexBinary(left, right, NativeCore.TryComplexAdd, ComplexAdd);

    /// <summary>尺寸要相同。同一張減自己，反轉換後是全黑。</summary>
    /// <remarks>頻譜相減。</remarks>
    /// <example>
    /// <code>Vizion.sub_fft (Complex1, Complex2, ComplexDiff)</code>
    /// </example>
    /// <param name="left">被減數。</param>
    /// <param name="right">減數。</param>
    /// <returns>逐點相減。</returns>
    public static ComplexBuffer SubFft(ComplexBuffer left, ComplexBuffer right) =>
        ComplexBinary(left, right, NativeCore.TryComplexSub, ComplexSub);

    /// <summary>尺寸要相同。在頻率相乘就是在空間做循環卷積。</summary>
    /// <remarks>頻譜相乘。</remarks>
    /// <example>
    /// <code>Vizion.mult_fft (Complex1, Complex2, ComplexProduct)</code>
    /// </example>
    /// <param name="left">第一張複數影像。</param>
    /// <param name="right">第二張複數影像。</param>
    /// <returns>逐點相乘。</returns>
    public static ComplexBuffer MultFft(ComplexBuffer left, ComplexBuffer right) =>
        ComplexBinary(left, right, NativeCore.TryComplexMul, ComplexMul);

    /// <summary>尺寸要相同。除數的絕對值接近 0 時，那個頻率的結果是 0。</summary>
    /// <remarks>頻譜相除。</remarks>
    /// <example>
    /// <code>Vizion.div_fft (Complex1, Complex2, ComplexQuotient)</code>
    /// </example>
    /// <param name="left">被除數。</param>
    /// <param name="right">除數。</param>
    /// <returns>逐點相除。</returns>
    public static ComplexBuffer DivFft(ComplexBuffer left, ComplexBuffer right) =>
        ComplexBinary(left, right, NativeCore.TryComplexDiv, ComplexDiv);

    /// <summary>dc 只留左上角。no_dc 把左上角清成 0。real 去掉虛部，imag 去掉實部。</summary>
    /// <remarks>遮罩頻譜。低頻還沒有 fftshift 時，直流就在 (0, 0)。</remarks>
    /// <example>
    /// <code>Vizion.mask_fft (Complex, 'dc', ComplexMasked)</code>
    /// </example>
    /// <param name="spectrum">複數頻譜。</param>
    /// <param name="mode">dc、no_dc、real 或 imag。</param>
    /// <returns>遮罩後的頻譜。</returns>
    public static ComplexBuffer MaskFft(ComplexBuffer spectrum, string mode)
    {
        if (spectrum == null)
            throw new ArgumentNullException(nameof(spectrum));
        int code = MaskCode(mode);
        var real = new double[spectrum.Real.Length];
        var imag = new double[spectrum.Imag.Length];
        if (!NativeCore.TryComplexMask(spectrum.Real, spectrum.Imag, real, imag, spectrum.Width, spectrum.Height, code))
            ComplexMask(spectrum.Real, spectrum.Imag, real, imag, spectrum.Width, spectrum.Height, code);
        return ComplexBuffer.Wrap(spectrum.Width, spectrum.Height, real, imag);
    }

    /// <summary>能量是所有頻率絕對值平方的平均，等於空間上每個像素平方的總和。常數灰階 g 的能量是 g 的平方乘上寬乘高。</summary>
    /// <remarks>計算頻譜能量。</remarks>
    /// <example>
    /// <code>Vizion.fft_energy (Complex, Energy)</code>
    /// </example>
    /// <param name="spectrum">複數頻譜。</param>
    /// <returns>能量。</returns>
    public static double FftEnergy(ComplexBuffer spectrum)
    {
        if (spectrum == null)
            throw new ArgumentNullException(nameof(spectrum));
        if (!NativeCore.TryComplexEnergy(spectrum.Real, spectrum.Imag, out double energy))
            energy = ComplexEnergy(spectrum.Real, spectrum.Imag);
        return energy;
    }

    static int ShiftCode(string mode)
    {
        if (mode == null)
            throw new ArgumentNullException(nameof(mode));
        string name = mode.Trim().Trim('\'').Trim('"').ToLowerInvariant();
        if (name == "center")
            return 1;
        if (name == "corner")
            return 0;
        throw new ArgumentException("Mode 只能是 center 或 corner");
    }

    static int MaskCode(string mode)
    {
        if (mode == null)
            throw new ArgumentNullException(nameof(mode));
        string name = mode.Trim().Trim('\'').Trim('"').ToLowerInvariant();
        return name switch
        {
            "dc" => 0,
            "no_dc" => 1,
            "real" => 2,
            "imag" => 3,
            _ => throw new ArgumentException("Mode 只能是 dc、no_dc、real 或 imag"),
        };
    }

    static void SameSpectrum(ComplexBuffer left, ComplexBuffer right)
    {
        if (left == null || right == null)
            throw new ArgumentNullException(nameof(left));
        if (left.Width != right.Width || left.Height != right.Height)
            throw new ArgumentException("兩張複數影像的尺寸要相同");
    }

    static ComplexBuffer ComplexUnary(ComplexBuffer spectrum, Action<double[], double[], double[], double[]> apply)
    {
        if (spectrum == null)
            throw new ArgumentNullException(nameof(spectrum));
        var real = new double[spectrum.Real.Length];
        var imag = new double[spectrum.Imag.Length];
        apply(spectrum.Real, spectrum.Imag, real, imag);
        return ComplexBuffer.Wrap(spectrum.Width, spectrum.Height, real, imag);
    }

    static ComplexBuffer ComplexBinary(
        ComplexBuffer left, ComplexBuffer right,
        Func<double[], double[], double[], double[], double[], double[], bool> native,
        Action<double[], double[], double[], double[], double[], double[]> scalar)
    {
        SameSpectrum(left, right);
        var real = new double[left.Real.Length];
        var imag = new double[left.Imag.Length];
        if (!native(left.Real, left.Imag, right.Real, right.Imag, real, imag))
            scalar(left.Real, left.Imag, right.Real, right.Imag, real, imag);
        return ComplexBuffer.Wrap(left.Width, left.Height, real, imag);
    }

    static void ComplexFill(double[] real, double[] imag, double re, double im)
    {
        for (int i = 0; i < real.Length; i++)
        {
            real[i] = re;
            imag[i] = im;
        }
    }

    static void ComplexScale(double[] real, double[] imag, double[] dstReal, double[] dstImag, double factor)
    {
        for (int i = 0; i < real.Length; i++)
        {
            dstReal[i] = real[i] * factor;
            dstImag[i] = imag[i] * factor;
        }
    }

    static void ComplexConj(double[] real, double[] imag, double[] dstReal, double[] dstImag)
    {
        for (int i = 0; i < real.Length; i++)
        {
            dstReal[i] = real[i];
            dstImag[i] = -imag[i];
        }
    }

    static void ComplexAdd(double[] leftReal, double[] leftImag, double[] rightReal, double[] rightImag, double[] dstReal, double[] dstImag)
    {
        for (int i = 0; i < leftReal.Length; i++)
        {
            dstReal[i] = leftReal[i] + rightReal[i];
            dstImag[i] = leftImag[i] + rightImag[i];
        }
    }

    static void ComplexSub(double[] leftReal, double[] leftImag, double[] rightReal, double[] rightImag, double[] dstReal, double[] dstImag)
    {
        for (int i = 0; i < leftReal.Length; i++)
        {
            dstReal[i] = leftReal[i] - rightReal[i];
            dstImag[i] = leftImag[i] - rightImag[i];
        }
    }

    static void ComplexMul(double[] leftReal, double[] leftImag, double[] rightReal, double[] rightImag, double[] dstReal, double[] dstImag)
    {
        for (int i = 0; i < leftReal.Length; i++)
        {
            double ar = leftReal[i];
            double ai = leftImag[i];
            double br = rightReal[i];
            double bi = rightImag[i];
            dstReal[i] = Fused(ar, br, -(ai * bi));
            dstImag[i] = Fused(ar, bi, ai * br);
        }
    }

    static void ComplexDiv(double[] leftReal, double[] leftImag, double[] rightReal, double[] rightImag, double[] dstReal, double[] dstImag)
    {
        for (int i = 0; i < leftReal.Length; i++)
        {
            double br = rightReal[i];
            double bi = rightImag[i];
            double absBr = Math.Abs(br);
            double absBi = Math.Abs(bi);
            if (!(Math.Sqrt(br * br + bi * bi) > 1e-12))
            {
                dstReal[i] = 0;
                dstImag[i] = 0;
                continue;
            }
            double ar = leftReal[i];
            double ai = leftImag[i];
            if (absBr >= absBi)
            {
                double rat = bi / br;
                double scl = 1.0 / (br + bi * rat);
                dstReal[i] = (ar + ai * rat) * scl;
                dstImag[i] = (ai - ar * rat) * scl;
            }
            else
            {
                double rat = br / bi;
                double scl = 1.0 / (bi + br * rat);
                dstReal[i] = (ar * rat + ai) * scl;
                dstImag[i] = (ai * rat - ar) * scl;
            }
        }
    }

    static void ComplexShift(double[] real, double[] imag, double[] dstReal, double[] dstImag, int width, int height, int center)
    {
        int rowShift = center != 0 ? height / 2 : -(height / 2);
        int colShift = center != 0 ? width / 2 : -(width / 2);
        for (int row = 0; row < height; row++)
        {
            int srcRow = SpectrumMod(row - rowShift, height);
            for (int col = 0; col < width; col++)
            {
                int srcCol = SpectrumMod(col - colShift, width);
                int dst = row * width + col;
                int src = srcRow * width + srcCol;
                dstReal[dst] = real[src];
                dstImag[dst] = imag[src];
            }
        }
    }

    static int SpectrumMod(int value, int size)
    {
        int wrapped = value % size;
        if (wrapped < 0)
            wrapped += size;
        return wrapped;
    }

    static void ComplexMask(double[] real, double[] imag, double[] dstReal, double[] dstImag, int width, int height, int mode)
    {
        int count = width * height;
        if (mode == 0)
        {
            Array.Clear(dstReal, 0, count);
            Array.Clear(dstImag, 0, count);
            dstReal[0] = real[0];
            dstImag[0] = imag[0];
            return;
        }
        if (mode == 1)
        {
            Array.Copy(real, dstReal, count);
            Array.Copy(imag, dstImag, count);
            dstReal[0] = 0;
            dstImag[0] = 0;
            return;
        }
        if (mode == 2)
        {
            Array.Copy(real, dstReal, count);
            Array.Clear(dstImag, 0, count);
            return;
        }
        Array.Clear(dstReal, 0, count);
        Array.Copy(imag, dstImag, count);
    }

    static double Fused(double a, double b, double c)
    {
        double product = a * b;
        const double split = 134217729.0;
        double ahScale = split * a;
        double ah = ahScale - (ahScale - a);
        double al = a - ah;
        double bhScale = split * b;
        double bh = bhScale - (bhScale - b);
        double bl = b - bh;
        double productError = ((ah * bh - product) + ah * bl + al * bh) + al * bl;
        double sum = product + c;
        double swapped = sum - product;
        double sumError = (product - (sum - swapped)) + (c - swapped);
        return (productError + sumError) + sum;
    }

    static double ComplexEnergy(double[] real, double[] imag)
    {
        double sum = 0;
        for (int i = 0; i < real.Length; i++)
        {
            double magnitude = Math.Sqrt(real[i] * real[i] + imag[i] * imag[i]);
            sum += magnitude * magnitude;
        }
        return sum / real.Length;
    }
}
