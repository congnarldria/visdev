namespace Vizion;

public static partial class Operators
{
    /// <summary>把影像、區域或輪廓畫到圖形視窗，不改已經算出的數值。顯示影像會清掉舊的疊加，顯示區域或輪廓會疊在目前影像上。</summary>
    /// <remarks>顯示到圖形視窗。參數 Object 是影像、區域或輪廓，由程式執行器確認這個變數存在。</remarks>
    /// <example>
    /// <code>Vizion.dev_display (Object)</code>
    /// </example>
    public static void DevDisplay()
    {
    }

    /// <summary>設定之後區域怎麼畫。margin 畫輪廓，fill 填色。不改已經算出的數值。</summary>
    /// <remarks>設定區域畫法。參數 DrawMode 是 margin 或 fill。這版的圖形視窗仍用目前的畫法。</remarks>
    /// <example>
    /// <code>Vizion.dev_set_draw ('margin')</code>
    /// </example>
    public static void DevSetDraw()
    {
    }
}
