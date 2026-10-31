namespace Vizion;

/// <summary>和 VisDev 畫面上同一組顏色，0xAARRGGBB。</summary>
public static class VizionColors
{
    public const uint Green = 0xFF00E05A;
    public const uint Red = 0xFFFF2A2A;
    public const uint Blue = 0xFF2F7DFF;
    public const uint Yellow = 0xFFFFE14A;
    public const uint Cyan = 0xFF2CE6E6;
    public const uint White = 0xFFFFFFFF;
    public const uint Black = 0xFF000000;

    public static uint WithAlpha(uint color, byte alpha) => (color & 0x00FFFFFFu) | ((uint)alpha << 24);
}
