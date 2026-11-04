namespace Vizion;

/// <summary>傅立葉頻譜。實部和虛部分開存，原點在左上角。</summary>
public sealed class ComplexBuffer
{
    public int Width { get; }
    public int Height { get; }
    public double[] Real { get; }
    public double[] Imag { get; }

    public ComplexBuffer(int width, int height, double[] real, double[] imag)
    {
        if (width < 1 || height < 1)
            throw new ArgumentOutOfRangeException(nameof(width), "寬和高至少要 1");
        if (real == null || imag == null)
            throw new ArgumentNullException(nameof(real));
        int count = checked(width * height);
        if (real.Length < count || imag.Length < count)
            throw new ArgumentException("複數緩衝比寬高還短");
        Width = width;
        Height = height;
        Real = new double[count];
        Imag = new double[count];
        Array.Copy(real, Real, count);
        Array.Copy(imag, Imag, count);
    }

    internal static ComplexBuffer Wrap(int width, int height, double[] real, double[] imag) =>
        new(width, height, real, imag, own: true);

    ComplexBuffer(int width, int height, double[] real, double[] imag, bool own)
    {
        Width = width;
        Height = height;
        Real = real;
        Imag = imag;
    }

    public override string ToString() => "複數影像  " + Width + "×" + Height;
}
