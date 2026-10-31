namespace Vizion;

/// <summary>縮放與平移。預設整張塞進視窗。放大後只保留看得到的那一塊。</summary>
public sealed class ViewState
{
    public bool Fitted { get; private set; } = true;
    public double Row1 { get; private set; }
    public double Column1 { get; private set; }
    public double Row2 { get; private set; }
    public double Column2 { get; private set; }

    public void Fit() => Fitted = true;

    public ScreenMap Map(int imageWidth, int imageHeight, int viewWidth, int viewHeight)
    {
        if (Fitted)
            return ScreenMap.Fit(imageWidth, imageHeight, viewWidth, viewHeight);
        return ScreenMap.Fill(viewWidth, viewHeight, Row1, Column1, Row2, Column2);
    }

    public void ZoomAt(double row, double column, double factor, int imageWidth, int imageHeight, int viewWidth, int viewHeight)
    {
        var map = Map(imageWidth, imageHeight, viewWidth, viewHeight);
        if (!map.HasImage || factor <= 0)
            return;
        double rowSpan = (map.Row2 - map.Row1) / factor;
        double columnSpan = (map.Column2 - map.Column1) / factor;
        rowSpan = Compat.Clamp(rowSpan, 4, Math.Max(4, imageHeight * 4));
        columnSpan = Compat.Clamp(columnSpan, 4, Math.Max(4, imageWidth * 4));
        double rowFraction = (row - map.Row1) / (map.Row2 - map.Row1);
        double columnFraction = (column - map.Column1) / (map.Column2 - map.Column1);
        SetPart(row - rowFraction * rowSpan, column - columnFraction * columnSpan, rowSpan, columnSpan, imageWidth, imageHeight);
    }

    public void PanPixels(int dx, int dy, ScreenMap map, int imageWidth, int imageHeight)
    {
        if (!map.HasImage)
            return;
        double dColumn = -dx * (map.Column2 - map.Column1) / map.DestWidth;
        double dRow = -dy * (map.Row2 - map.Row1) / map.DestHeight;
        SetPart(map.Row1 + dRow, map.Column1 + dColumn, map.Row2 - map.Row1, map.Column2 - map.Column1, imageWidth, imageHeight);
    }

    private void SetPart(double row1, double column1, double rowSpan, double columnSpan, int imageWidth, int imageHeight)
    {
        if (row1 < 0)
            row1 = 0;
        if (column1 < 0)
            column1 = 0;
        if (row1 + rowSpan > imageHeight)
            row1 = Math.Max(0, imageHeight - rowSpan);
        if (column1 + columnSpan > imageWidth)
            column1 = Math.Max(0, imageWidth - columnSpan);
        Row1 = row1;
        Column1 = column1;
        Row2 = row1 + rowSpan;
        Column2 = column1 + columnSpan;
        Fitted = false;
    }
}
