namespace Vizion;

/// <summary>和 OpenCV COLOR_BGR2GRAY 同一個整數式：((R*9798 + G*19235 + B*3735) + 16384) &gt;&gt; 15。</summary>
public static class GrayConvert
{
    public static int Bgr(byte b, byte g, byte r) => (r * 9798 + g * 19235 + b * 3735 + 16384) >> 15;
}
