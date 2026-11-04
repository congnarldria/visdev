using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace Photonix;

enum ProgramRailKind { Gutter, Times }

/// <summary>程式左側的行號和斷點，或右側的每一行時間。垂直位置跟文字方塊對齊。</summary>
sealed class ProgramRail : FrameworkElement
{
    static readonly Brush GutterBg = Freeze(Color.FromRgb(0x17, 0x1B, 0x21));
    static readonly Brush TimeBg = Freeze(Color.FromRgb(0x0E, 0x11, 0x16));
    static readonly Brush Number = Freeze(Color.FromRgb(0x70, 0x78, 0x88));
    static readonly Brush CurrentText = Freeze(Color.FromRgb(0xE7, 0xFF, 0xF3));
    static readonly Brush CurrentBg = Freeze(Color.FromArgb(36, 0, 224, 90));
    static readonly Brush AimBg = Freeze(Color.FromArgb(20, 0, 224, 90));
    static readonly Brush Arrow = Freeze(Color.FromRgb(0x00, 0xE0, 0x5A));
    static readonly Brush AimArrow = Freeze(Color.FromArgb(140, 0, 224, 90));
    static readonly Brush Break = Freeze(Color.FromRgb(0xFF, 0x5D, 0x4A));
    static readonly Brush Time = Freeze(Color.FromRgb(0x8F, 0xD6, 0xA8));

    readonly List<LineRow> Rows = [];
    bool Dragging;

    public ProgramRailKind Kind { get; set; }
    public TextBox? Editor { get; set; }
    public int CurrentLine { get; set; }
    public int AimLine { get; set; }
    public HashSet<int> Breakpoints { get; set; } = [];
    public Dictionary<int, (double Ms, int Count)> Times { get; set; } = [];

    public event EventHandler<int>? LineClick;
    public event EventHandler<int>? ArrowMoved;

    public ProgramRail()
    {
        SnapsToDevicePixels = true;
    }

    public void Attach(TextBox editor)
    {
        Editor = editor;
        editor.TextChanged += (_, _) => InvalidateVisual();
        editor.SizeChanged += (_, _) => InvalidateVisual();
        editor.Loaded += (_, _) => HookScroll();
        if (editor.IsLoaded)
            HookScroll();
    }

    void HookScroll()
    {
        if (Editor?.Template.FindName("PART_ContentHost", Editor) is System.Windows.Controls.ScrollViewer scroll)
            scroll.ScrollChanged += (_, _) => InvalidateVisual();
    }

    protected override void OnRender(DrawingContext drawingContext)
    {
        Rebuild();
        drawingContext.DrawRectangle(Kind == ProgramRailKind.Gutter ? GutterBg : TimeBg, null, new Rect(0, 0, ActualWidth, ActualHeight));
        if (Editor == null)
            return;
        double dpi = VisualTreeHelper.GetDpi(this).PixelsPerDip;
        var typeface = new Typeface(new FontFamily("Consolas"), FontStyles.Normal, FontWeights.Normal, FontStretches.Normal);
        double size = Kind == ProgramRailKind.Times ? 11 : 12;
        foreach (LineRow row in Rows)
        {
            bool current = row.Line == CurrentLine;
            bool aim = row.Line == AimLine && AimLine != CurrentLine;
            if (current)
                drawingContext.DrawRectangle(CurrentBg, null, new Rect(0, row.Y, ActualWidth, row.Height));
            else if (aim)
                drawingContext.DrawRectangle(AimBg, null, new Rect(0, row.Y, ActualWidth, row.Height));
            if (Kind == ProgramRailKind.Gutter)
                DrawGutter(drawingContext, row, current, aim, typeface, size, dpi);
            else if (Times.TryGetValue(row.Line, out var time))
                DrawTime(drawingContext, ProgramCounter.FormatTime(time.Ms, time.Count), row, current ? CurrentText : Time, typeface, size, dpi);
        }
    }

    void DrawGutter(DrawingContext drawingContext, LineRow row, bool current, bool aim, Typeface typeface, double size, double dpi)
    {
        if (current || aim)
            DrawArrow(drawingContext, 1, row.Y, row.Height, aim ? AimArrow : Arrow);
        var number = new FormattedText(row.Line.ToString(), CultureInfo.CurrentUICulture, FlowDirection.LeftToRight, typeface, size, current ? CurrentText : Number, dpi);
        double x = ActualWidth - 6 - number.WidthIncludingTrailingWhitespace;
        drawingContext.DrawText(number, new Point(x, row.Y + Math.Max(0, (row.Height - number.Height) / 2)));
        if (Breakpoints.Contains(row.Line))
            drawingContext.DrawEllipse(Break, null, new Point(x - 8, row.Y + row.Height / 2), 3.5, 3.5);
    }

    static void DrawArrow(DrawingContext drawingContext, double x, double y, double height, Brush brush)
    {
        double mid = y + height / 2;
        var geometry = new StreamGeometry();
        using (StreamGeometryContext context = geometry.Open())
        {
            context.BeginFigure(new Point(x + 2, mid - 6), true, true);
            context.LineTo(new Point(x + 13, mid), true, false);
            context.LineTo(new Point(x + 2, mid + 6), true, false);
        }
        geometry.Freeze();
        drawingContext.DrawGeometry(brush, null, geometry);
    }

    void DrawTime(DrawingContext drawingContext, string text, LineRow row, Brush brush, Typeface typeface, double size, double dpi)
    {
        var formatted = new FormattedText(text, CultureInfo.CurrentUICulture, FlowDirection.LeftToRight, typeface, size, brush, dpi);
        double x = ActualWidth - 8 - formatted.WidthIncludingTrailingWhitespace;
        drawingContext.DrawText(formatted, new Point(x, row.Y + Math.Max(0, (row.Height - formatted.Height) / 2)));
    }

    void Rebuild()
    {
        Rows.Clear();
        if (Editor == null)
            return;
        string text = Editor.Text ?? "";
        int index = 0;
        int line = 1;
        while (true)
        {
            int newline = index >= text.Length ? -1 : text.IndexOf('\n', index);
            int end = newline < 0 ? text.Length : newline;
            int caret = Math.Min(index, text.Length);
            Rect rect = Editor.GetRectFromCharacterIndex(caret);
            double bottom = rect.Y + Math.Max(rect.Height, 1);
            if (newline >= 0)
            {
                Rect next = Editor.GetRectFromCharacterIndex(Math.Min(newline + 1, text.Length));
                if (next.Y > rect.Y)
                    bottom = next.Y;
            }
            Rows.Add(new LineRow(line, rect.Y, Math.Max(1, bottom - rect.Y)));
            if (newline < 0)
                break;
            index = newline + 1;
            line++;
            if (index > text.Length)
                break;
        }
    }

    int LineFromY(double y)
    {
        foreach (LineRow row in Rows)
        {
            if (y >= row.Y && y < row.Y + row.Height)
                return row.Line;
        }
        if (Rows.Count == 0)
            return 1;
        if (y < Rows[0].Y)
            return Rows[0].Line;
        return Rows[^1].Line;
    }

    protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
    {
        if (Kind != ProgramRailKind.Gutter)
            return;
        Rebuild();
        Point point = e.GetPosition(this);
        int line = LineFromY(point.Y);
        if (line == CurrentLine && point.X < 18)
        {
            Dragging = true;
            AimLine = line;
            CaptureMouse();
            Cursor = Cursors.Hand;
            InvalidateVisual();
            e.Handled = true;
        }
        base.OnMouseLeftButtonDown(e);
    }

    protected override void OnMouseMove(MouseEventArgs e)
    {
        if (!Dragging)
        {
            if (Kind == ProgramRailKind.Gutter)
            {
                Rebuild();
                Point point = e.GetPosition(this);
                Cursor = LineFromY(point.Y) == CurrentLine && point.X < 18 ? Cursors.Hand : Cursors.Arrow;
            }
            return;
        }
        int line = LineFromY(e.GetPosition(this).Y);
        if (line != AimLine)
        {
            AimLine = line;
            InvalidateVisual();
        }
        e.Handled = true;
    }

    protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e)
    {
        if (Kind != ProgramRailKind.Gutter)
            return;
        Rebuild();
        int line = LineFromY(e.GetPosition(this).Y);
        if (Dragging)
        {
            Dragging = false;
            ReleaseMouseCapture();
            Cursor = Cursors.Arrow;
            int target = AimLine == 0 ? line : AimLine;
            AimLine = 0;
            InvalidateVisual();
            if (target != CurrentLine)
                ArrowMoved?.Invoke(this, target);
            e.Handled = true;
            return;
        }
        LineClick?.Invoke(this, line);
        e.Handled = true;
    }

    static Brush Freeze(Color color)
    {
        var brush = new SolidColorBrush(color);
        brush.Freeze();
        return brush;
    }

    readonly record struct LineRow(int Line, double Y, double Height);
}
