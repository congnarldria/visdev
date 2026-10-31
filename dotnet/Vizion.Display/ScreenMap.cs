namespace Vizion;

/// <summary>影像上的一塊範圍畫到控制項的哪個矩形。列、行的下界不含。</summary>
public readonly struct ScreenMap
{
    public int DestX { get; }
    public int DestY { get; }
    public int DestWidth { get; }
    public int DestHeight { get; }
    public double Row1 { get; }
    public double Column1 { get; }
    public double Row2 { get; }
    public double Column2 { get; }

    public ScreenMap(int destX, int destY, int destWidth, int destHeight, double row1, double column1, double row2, double column2)
    {
        DestX = destX;
        DestY = destY;
        DestWidth = destWidth;
        DestHeight = destHeight;
        Row1 = row1;
        Column1 = column1;
        Row2 = row2;
        Column2 = column2;
    }

    public bool HasImage => DestWidth > 0 && DestHeight > 0 && Row2 > Row1 && Column2 > Column1;

    public static ScreenMap Fit(int imageWidth, int imageHeight, int viewWidth, int viewHeight)
    {
        if (imageWidth <= 0 || imageHeight <= 0 || viewWidth <= 0 || viewHeight <= 0)
            return default;
        double scale = Math.Min((double)viewWidth / imageWidth, (double)viewHeight / imageHeight);
        int destWidth = Math.Max(1, (int)Math.Round(imageWidth * scale));
        int destHeight = Math.Max(1, (int)Math.Round(imageHeight * scale));
        int destX = (viewWidth - destWidth) / 2;
        int destY = (viewHeight - destHeight) / 2;
        return new ScreenMap(destX, destY, destWidth, destHeight, 0, 0, imageHeight, imageWidth);
    }

    public static ScreenMap Fill(int viewWidth, int viewHeight, double row1, double column1, double row2, double column2)
    {
        if (viewWidth <= 0 || viewHeight <= 0)
            return default;
        return new ScreenMap(0, 0, viewWidth, viewHeight, row1, column1, row2, column2);
    }

    public bool TryImage(int x, int y, out double row, out double column)
    {
        row = column = 0;
        if (!HasImage || x < DestX || y < DestY || x >= DestX + DestWidth || y >= DestY + DestHeight)
            return false;
        double fx = (x - DestX + 0.5) / DestWidth;
        double fy = (y - DestY + 0.5) / DestHeight;
        column = Column1 + fx * (Column2 - Column1);
        row = Row1 + fy * (Row2 - Row1);
        return true;
    }

    public (float X, float Y) ToScreen(double row, double column)
    {
        float x = DestX + (float)((column - Column1) / (Column2 - Column1) * DestWidth);
        float y = DestY + (float)((row - Row1) / (Row2 - Row1) * DestHeight);
        return (x, y);
    }
}
