namespace Vizion;

/// <summary>
/// 把影像畫進 BGRA 緩衝。放大時只讀視窗裡的像素；縮小時每個來源像素只讀一次。
/// </summary>
public static class FrameComposer
{
    public const uint Background = 0xFF1E1E1E;

    public static unsafe void Render(ImageFrame frame, ScreenMap map, Overlay[] overlays, byte[] destination, int destinationWidth, int destinationHeight)
    {
        if (destinationWidth <= 0 || destinationHeight <= 0)
            return;
        if (overlays == null)
            overlays = Array.Empty<Overlay>();
        int pixels = destinationWidth * destinationHeight;
        if (destination.Length < pixels * 4)
            throw new ArgumentException("目的緩衝比畫面還小");
        fixed (byte* pinned = destination)
        {
            uint* bgra = (uint*)pinned;
            for (int index = 0; index < pixels; index++)
                bgra[index] = Background;
            if (map.HasImage)
                PaintImage(frame, map, bgra, destinationWidth);
            for (int index = 0; index < overlays.Length; index++)
                PaintOverlay(overlays[index], map, bgra, destinationWidth, destinationHeight);
        }
    }

    private static unsafe void PaintImage(ImageFrame frame, ScreenMap map, uint* bgra, int destinationWidth)
    {
        double rowSpan = map.Row2 - map.Row1;
        double columnSpan = map.Column2 - map.Column1;
        bool oneToOne = map.DestWidth == (int)columnSpan && map.DestHeight == (int)rowSpan
            && Math.Abs(map.Row1 - Math.Round(map.Row1)) < 1e-6
            && Math.Abs(map.Column1 - Math.Round(map.Column1)) < 1e-6;
        if (oneToOne)
        {
            BlitOne(frame, (int)Math.Round(map.Row1), (int)Math.Round(map.Column1), map, bgra, destinationWidth);
            return;
        }
        for (int y = 0; y < map.DestHeight; y++)
        {
            int row0 = (int)Math.Floor(map.Row1 + y * rowSpan / map.DestHeight);
            int row1 = (int)Math.Floor(map.Row1 + (y + 1) * rowSpan / map.DestHeight);
            if (row1 <= row0)
                row1 = row0 + 1;
            row0 = Compat.Clamp(row0, 0, frame.Height);
            row1 = Compat.Clamp(row1, row0, frame.Height);
            if (row0 == row1)
                continue;
            int destRow = map.DestY + y;
            for (int x = 0; x < map.DestWidth; x++)
            {
                int col0 = (int)Math.Floor(map.Column1 + x * columnSpan / map.DestWidth);
                int col1 = (int)Math.Floor(map.Column1 + (x + 1) * columnSpan / map.DestWidth);
                if (col1 <= col0)
                    col1 = col0 + 1;
                col0 = Compat.Clamp(col0, 0, frame.Width);
                col1 = Compat.Clamp(col1, col0, frame.Width);
                if (col0 == col1)
                    continue;
                bgra[destRow * destinationWidth + map.DestX + x] = Average(frame, row0, row1, col0, col1);
            }
        }
    }

    private static unsafe void BlitOne(ImageFrame frame, int row0, int column0, ScreenMap map, uint* bgra, int destinationWidth)
    {
        for (int y = 0; y < map.DestHeight; y++)
        {
            int row = row0 + y;
            if ((uint)row >= (uint)frame.Height)
                continue;
            int dest = (map.DestY + y) * destinationWidth + map.DestX;
            for (int x = 0; x < map.DestWidth; x++)
            {
                int column = column0 + x;
                if ((uint)column >= (uint)frame.Width)
                    continue;
                bgra[dest + x] = Pack(frame, row, column);
            }
        }
    }

    private static uint Average(ImageFrame frame, int row0, int row1, int col0, int col1)
    {
        long b = 0, g = 0, r = 0, n = 0;
        int channels = frame.Channels;
        for (int row = row0; row < row1; row++)
        {
            int index = row * frame.Stride + col0 * channels;
            for (int column = col0; column < col1; column++)
            {
                if (channels == 1)
                {
                    int gray = frame.Pixels[index];
                    b += gray;
                    g += gray;
                    r += gray;
                    index += 1;
                }
                else if (frame.Layout == PixelLayout.Bgr)
                {
                    b += frame.Pixels[index];
                    g += frame.Pixels[index + 1];
                    r += frame.Pixels[index + 2];
                    index += 3;
                }
                else
                {
                    r += frame.Pixels[index];
                    g += frame.Pixels[index + 1];
                    b += frame.Pixels[index + 2];
                    index += 3;
                }
                n++;
            }
        }
        if (n == 0)
            return Background;
        return 0xFF000000u | (uint)(r / n) << 16 | (uint)(g / n) << 8 | (uint)(b / n);
    }

    private static uint Pack(ImageFrame frame, int row, int column)
    {
        int index = row * frame.Stride + column * frame.Channels;
        if (frame.Channels == 1)
        {
            uint gray = frame.Pixels[index];
            return 0xFF000000u | (gray << 16) | (gray << 8) | gray;
        }
        uint b, g, r;
        if (frame.Layout == PixelLayout.Bgr)
        {
            b = frame.Pixels[index];
            g = frame.Pixels[index + 1];
            r = frame.Pixels[index + 2];
        }
        else
        {
            r = frame.Pixels[index];
            g = frame.Pixels[index + 1];
            b = frame.Pixels[index + 2];
        }
        return 0xFF000000u | (r << 16) | (g << 8) | b;
    }

    private static unsafe void PaintOverlay(Overlay overlay, ScreenMap map, uint* bgra, int destinationWidth, int destinationHeight)
    {
        if (!map.HasImage)
            return;
        switch (overlay.Kind)
        {
            case OverlayKind.Rectangle:
                StrokeRect(map, bgra, destinationWidth, destinationHeight, overlay.Row1, overlay.Column1, overlay.Row2, overlay.Column2, overlay.Color, overlay.Filled);
                break;
            case OverlayKind.Cross:
                StrokeRect(map, bgra, destinationWidth, destinationHeight, overlay.Row1 - overlay.Size, overlay.Column1, overlay.Row1 + overlay.Size, overlay.Column1, overlay.Color, false);
                StrokeRect(map, bgra, destinationWidth, destinationHeight, overlay.Row1, overlay.Column1 - overlay.Size, overlay.Row1, overlay.Column1 + overlay.Size, overlay.Color, false);
                break;
            case OverlayKind.Polyline:
                for (int index = 1; index < overlay.Rows.Length; index++)
                {
                    var from = map.ToScreen(overlay.Rows[index - 1], overlay.Columns[index - 1]);
                    var to = map.ToScreen(overlay.Rows[index], overlay.Columns[index]);
                    Line(bgra, destinationWidth, destinationHeight, (int)Math.Round(from.X), (int)Math.Round(from.Y), (int)Math.Round(to.X), (int)Math.Round(to.Y), overlay.Color);
                }
                break;
            case OverlayKind.Runs:
                PaintRuns(overlay, map, bgra, destinationWidth, destinationHeight);
                break;
        }
    }

    private static unsafe void PaintRuns(Overlay overlay, ScreenMap map, uint* bgra, int destinationWidth, int destinationHeight)
    {
        for (int index = 0; index < overlay.RunRows.Length; index++)
        {
            double row = overlay.RunRows[index] + 0.5;
            if (row < map.Row1 || row >= map.Row2)
                continue;
            int y = (int)map.ToScreen(row, map.Column1).Y;
            int x0 = (int)Math.Round(map.ToScreen(row, overlay.RunColumns[index]).X);
            int x1 = (int)Math.Round(map.ToScreen(row, overlay.RunColumns[index] + overlay.RunLengths[index]).X);
            if (x1 < x0)
                (x0, x1) = (x1, x0);
            FillSpan(bgra, destinationWidth, destinationHeight, x0, x1, y, overlay.Color);
        }
    }

    private static unsafe void StrokeRect(ScreenMap map, uint* bgra, int destinationWidth, int destinationHeight, double row1, double column1, double row2, double column2, uint color, bool filled)
    {
        var a = map.ToScreen(row1, column1);
        var b = map.ToScreen(row2, column2);
        int x0 = (int)Math.Round(Math.Min(a.X, b.X));
        int x1 = (int)Math.Round(Math.Max(a.X, b.X));
        int y0 = (int)Math.Round(Math.Min(a.Y, b.Y));
        int y1 = (int)Math.Round(Math.Max(a.Y, b.Y));
        if (filled)
        {
            for (int y = y0; y <= y1; y++)
                FillSpan(bgra, destinationWidth, destinationHeight, x0, x1 + 1, y, color);
            return;
        }
        Line(bgra, destinationWidth, destinationHeight, x0, y0, x1, y0, color);
        Line(bgra, destinationWidth, destinationHeight, x1, y0, x1, y1, color);
        Line(bgra, destinationWidth, destinationHeight, x1, y1, x0, y1, color);
        Line(bgra, destinationWidth, destinationHeight, x0, y1, x0, y0, color);
    }

    private static unsafe void FillSpan(uint* bgra, int destinationWidth, int destinationHeight, int x0, int x1, int y, uint color)
    {
        if ((uint)y >= (uint)destinationHeight)
            return;
        if (x0 < 0)
            x0 = 0;
        if (x1 > destinationWidth)
            x1 = destinationWidth;
        int row = y * destinationWidth;
        for (int x = x0; x < x1; x++)
            bgra[row + x] = Blend(bgra[row + x], color);
    }

    private static unsafe void Line(uint* bgra, int destinationWidth, int destinationHeight, int x0, int y0, int x1, int y1, uint color)
    {
        int dx = Math.Abs(x1 - x0);
        int dy = Math.Abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1;
        int sy = y0 < y1 ? 1 : -1;
        int err = dx - dy;
        while (true)
        {
            if ((uint)x0 < (uint)destinationWidth && (uint)y0 < (uint)destinationHeight)
                bgra[y0 * destinationWidth + x0] = Blend(bgra[y0 * destinationWidth + x0], color);
            if (x0 == x1 && y0 == y1)
                break;
            int e2 = err * 2;
            if (e2 > -dy)
            {
                err -= dy;
                x0 += sx;
            }
            if (e2 < dx)
            {
                err += dx;
                y0 += sy;
            }
        }
    }

    private static uint Blend(uint below, uint above)
    {
        uint alpha = above >> 24;
        if (alpha >= 255)
            return above | 0xFF000000u;
        if (alpha == 0)
            return below;
        uint inv = 255 - alpha;
        uint b = ((below & 0xFF) * inv + (above & 0xFF) * alpha) / 255;
        uint g = (((below >> 8) & 0xFF) * inv + ((above >> 8) & 0xFF) * alpha) / 255;
        uint r = (((below >> 16) & 0xFF) * inv + ((above >> 16) & 0xFF) * alpha) / 255;
        return 0xFF000000u | (r << 16) | (g << 8) | b;
    }
}
