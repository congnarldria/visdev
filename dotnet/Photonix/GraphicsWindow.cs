using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Shapes;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Vizion;

namespace Photonix;

/// <summary>跟主視窗同一張圖的獨立圖形視窗。可以點選、手抓和縮放。</summary>
sealed class GraphicsWindow : Window
{
    readonly Grid Stage = new() { ClipToBounds = true, Background = Brush(0x0E, 0x11, 0x16) };
    readonly Image Picture = new() { Stretch = Stretch.None, IsHitTestVisible = false, Visibility = Visibility.Collapsed };
    readonly TextBlock Empty = new()
    {
        Text = "圖形視窗是空的。\n跟主視窗同一張圖。",
        Foreground = Brush(0x8B, 0x94, 0x9E),
        HorizontalAlignment = HorizontalAlignment.Center,
        VerticalAlignment = VerticalAlignment.Center,
        TextAlignment = TextAlignment.Center,
        IsHitTestVisible = false,
    };
    readonly TextBlock ZoomText = new() { Text = "100%", Foreground = Brush(0x8B, 0x94, 0x9E), VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(8, 0, 0, 0) };
    readonly TextBlock Note = new() { Text = "跟主視窗同一張圖。點選、手抓、拉框放大。", Foreground = Brush(0x8B, 0x94, 0x9E), VerticalAlignment = VerticalAlignment.Center };
    readonly Line CrossV = new() { Stroke = Brush(0x59, 0xFF, 0xFF, 0xFF), StrokeThickness = 1, Visibility = Visibility.Collapsed, IsHitTestVisible = false };
    readonly Line CrossH = new() { Stroke = Brush(0x59, 0xFF, 0xFF, 0xFF), StrokeThickness = 1, Visibility = Visibility.Collapsed, IsHitTestVisible = false };
    readonly Rectangle ZoomBox = new() { Stroke = Brush(0xFF, 0xE1, 0x4A), Fill = Brush(0x1F, 0xFF, 0xE1, 0x4A), StrokeThickness = 1, Visibility = Visibility.Collapsed, IsHitTestVisible = false };
    readonly Border Probe = new() { Visibility = Visibility.Collapsed, Background = Brush(0xF0, 0x14, 0x17, 0x1C), BorderBrush = Brush(0x2D, 0x33, 0x3B), BorderThickness = new Thickness(1), Padding = new Thickness(6, 3, 6, 3), IsHitTestVisible = false };
    readonly TextBlock ProbeText = new() { Foreground = Brush(0xE6, 0xED, 0xF3) };
    readonly Button ModeSelect;
    readonly Button ModePan;
    readonly Button ModeZoom;
    readonly Brush Off = Brush(0x21, 0x26, 0x2D);
    readonly Brush On = Brush(0x3D, 0x8B, 0xFD);

    ImageBuffer? PictureData;
    ImageFrame? Frame;
    Overlay[] Shapes = [];
    double Zoom = 1;
    double CamX;
    double CamY;
    bool PendingFit = true;
    bool Dragging;
    Point DragStart;
    double DragCamX;
    double DragCamY;
    double DragCol;
    double DragRow;
    ModeKind Mode = ModeKind.Select;

    enum ModeKind { Select, Pan, Zoom }

    public GraphicsWindow()
    {
        Title = "圖形視窗";
        Width = 960;
        Height = 720;
        MinWidth = 480;
        MinHeight = 320;
        Background = Brush(0x14, 0x17, 0x1C);
        Foreground = Brush(0xE6, 0xED, 0xF3);
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        ModeSelect = ModeButton("點選", ModeKind.Select);
        ModePan = ModeButton("手抓", ModeKind.Pan);
        ModeZoom = ModeButton("放大", ModeKind.Zoom);
        var tools = new WrapPanel();
        tools.Children.Add(ModeSelect);
        tools.Children.Add(ModePan);
        tools.Children.Add(ModeZoom);
        tools.Children.Add(Tool("符合", (_, _) => Fit()));
        tools.Children.Add(Tool("1:1", (_, _) => One()));
        tools.Children.Add(Tool("−", (_, _) => ZoomAt(Stage.ActualWidth / 2, Stage.ActualHeight / 2, 0.8)));
        tools.Children.Add(Tool("+", (_, _) => ZoomAt(Stage.ActualWidth / 2, Stage.ActualHeight / 2, 1.25)));
        tools.Children.Add(ZoomText);
        var head = new DockPanel { Margin = new Thickness(8, 6, 8, 6) };
        DockPanel.SetDock(Note, Dock.Right);
        head.Children.Add(Note);
        head.Children.Add(new TextBlock { Text = "圖形視窗", FontWeight = FontWeights.Bold, VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(0, 0, 8, 0) });
        head.Children.Add(tools);
        var bar = new Border { Child = head, Background = Brush(0x1C, 0x21, 0x28), BorderBrush = Brush(0x2D, 0x33, 0x3B), BorderThickness = new Thickness(0, 0, 0, 1) };
        Probe.Child = ProbeText;
        var overlay = new Canvas { IsHitTestVisible = false };
        overlay.Children.Add(CrossV);
        overlay.Children.Add(CrossH);
        overlay.Children.Add(ZoomBox);
        overlay.Children.Add(Probe);
        var host = new Grid();
        host.Children.Add(Picture);
        host.Children.Add(overlay);
        host.Children.Add(Empty);
        Stage.Children.Add(host);
        Stage.MouseWheel += OnWheel;
        Stage.MouseLeftButtonDown += OnDown;
        Stage.MouseMove += OnMove;
        Stage.MouseLeftButtonUp += OnUp;
        Stage.MouseLeave += OnLeave;
        Stage.SizeChanged += (_, _) =>
        {
            if (PendingFit)
                Fit();
            else
                Present();
        };
        var root = new DockPanel();
        DockPanel.SetDock(bar, Dock.Top);
        root.Children.Add(bar);
        root.Children.Add(Stage);
        Content = root;
        ApplyMode(ModeKind.Select);
    }

    public void UpdateView(ImageBuffer? image, ImageFrame? frame, Overlay[]? shapes)
    {
        bool changed = image?.Width != PictureData?.Width || image?.Height != PictureData?.Height;
        PictureData = image;
        Frame = frame;
        Shapes = shapes ?? [];
        Empty.Visibility = image == null ? Visibility.Visible : Visibility.Collapsed;
        if (image == null)
        {
            Picture.Source = null;
            Picture.Visibility = Visibility.Collapsed;
            return;
        }
        if (changed || PendingFit)
            Fit();
        else
            Present();
    }

    Button ModeButton(string text, ModeKind mode)
    {
        var button = Tool(text, (_, _) => ApplyMode(mode));
        button.Tag = mode;
        return button;
    }

    static Button Tool(string text, RoutedEventHandler click)
    {
        var button = new Button
        {
            Content = text,
            Margin = new Thickness(0, 0, 4, 0),
            Padding = new Thickness(8, 3, 8, 3),
            Background = Brush(0x21, 0x26, 0x2D),
            Foreground = Brush(0xE6, 0xED, 0xF3),
            BorderBrush = Brush(0x2D, 0x33, 0x3B),
        };
        button.Click += click;
        return button;
    }

    void ApplyMode(ModeKind mode)
    {
        Mode = mode;
        ModeSelect.Background = mode == ModeKind.Select ? On : Off;
        ModePan.Background = mode == ModeKind.Pan ? On : Off;
        ModeZoom.Background = mode == ModeKind.Zoom ? On : Off;
        Stage.Cursor = mode switch
        {
            ModeKind.Pan => Cursors.Hand,
            ModeKind.Zoom => Cursors.Cross,
            _ => Cursors.Arrow,
        };
    }

    void Fit()
    {
        if (PictureData == null)
            return;
        double viewW = Stage.ActualWidth;
        double viewH = Stage.ActualHeight;
        if (viewW < 2 || viewH < 2)
        {
            PendingFit = true;
            return;
        }
        PendingFit = false;
        const double pad = 18;
        double scale = Math.Max(0.02, Math.Min((viewW - pad * 2) / PictureData.Width, (viewH - pad * 2) / PictureData.Height));
        ApplyView(scale, (viewW - PictureData.Width * scale) / 2, (viewH - PictureData.Height * scale) / 2);
    }

    void One()
    {
        if (PictureData == null)
            return;
        ApplyView(1, (Stage.ActualWidth - PictureData.Width) / 2, (Stage.ActualHeight - PictureData.Height) / 2);
    }

    void ZoomAt(double sx, double sy, double factor)
    {
        if (PictureData == null || Zoom <= 0)
            return;
        double next = Math.Clamp(Zoom * factor, 0.05, 48);
        double k = next / Zoom;
        ApplyView(next, sx - (sx - CamX) * k, sy - (sy - CamY) * k);
    }

    void ZoomToRect(double col0, double row0, double col1, double row1)
    {
        double viewW = Stage.ActualWidth;
        double viewH = Stage.ActualHeight;
        double left = Math.Min(col0, col1);
        double top = Math.Min(row0, row1);
        double width = Math.Max(1, Math.Abs(col1 - col0));
        double height = Math.Max(1, Math.Abs(row1 - row0));
        const double pad = 8;
        double scale = Math.Clamp(Math.Min((viewW - pad * 2) / width, (viewH - pad * 2) / height), 0.05, 48);
        ApplyView(scale, (viewW - width * scale) / 2 - left * scale, (viewH - height * scale) / 2 - top * scale);
    }

    void ApplyView(double scale, double x, double y)
    {
        Zoom = scale;
        CamX = x;
        CamY = y;
        ZoomText.Text = (scale * 100).ToString("0", CultureInfo.InvariantCulture) + "%";
        Present();
    }

    void Present()
    {
        if (PictureData == null || Frame == null)
            return;
        int viewW = (int)Math.Round(Stage.ActualWidth);
        int viewH = (int)Math.Round(Stage.ActualHeight);
        if (viewW < 2 || viewH < 2)
            return;
        var map = new ScreenMap(
            0, 0, viewW, viewH,
            (0 - CamY) / Zoom,
            (0 - CamX) / Zoom,
            (viewH - CamY) / Zoom,
            (viewW - CamX) / Zoom);
        var pixels = new byte[viewW * viewH * 4];
        FrameComposer.Render(Frame, map, Shapes, pixels, viewW, viewH);
        var source = BitmapSource.Create(viewW, viewH, 96, 96, PixelFormats.Bgra32, null, pixels, viewW * 4);
        source.Freeze();
        Picture.Source = source;
        Picture.Visibility = Visibility.Visible;
    }

    (double col, double row) ImagePoint(Point screen) => ((screen.X - CamX) / Zoom, (screen.Y - CamY) / Zoom);

    void OnWheel(object sender, MouseWheelEventArgs e)
    {
        if (PictureData == null)
            return;
        Point at = e.GetPosition(Stage);
        ZoomAt(at.X, at.Y, e.Delta > 0 ? 1.12 : 1 / 1.12);
        UpdateProbe(at);
        e.Handled = true;
    }

    void OnDown(object sender, MouseButtonEventArgs e)
    {
        if (PictureData == null)
            return;
        Stage.CaptureMouse();
        Dragging = true;
        DragStart = e.GetPosition(Stage);
        DragCamX = CamX;
        DragCamY = CamY;
        (DragCol, DragRow) = ImagePoint(DragStart);
        if (Mode == ModeKind.Zoom)
            ShowZoom(DragStart, DragStart);
    }

    void OnMove(object sender, MouseEventArgs e)
    {
        Point at = e.GetPosition(Stage);
        if (Dragging && PictureData != null && Mode == ModeKind.Pan)
            ApplyView(Zoom, DragCamX + at.X - DragStart.X, DragCamY + at.Y - DragStart.Y);
        else if (Dragging && Mode == ModeKind.Zoom)
            ShowZoom(DragStart, at);
        UpdateProbe(at);
    }

    void OnUp(object sender, MouseButtonEventArgs e)
    {
        if (!Dragging)
            return;
        Dragging = false;
        Stage.ReleaseMouseCapture();
        ZoomBox.Visibility = Visibility.Collapsed;
        if (PictureData == null)
            return;
        Point at = e.GetPosition(Stage);
        (double col, double row) = ImagePoint(at);
        double dx = at.X - DragStart.X;
        double dy = at.Y - DragStart.Y;
        bool moved = dx * dx + dy * dy > 16;
        if (Mode == ModeKind.Zoom)
        {
            if (moved)
                ZoomToRect(DragCol, DragRow, col, row);
            else
                ZoomAt(at.X, at.Y, 1.4);
        }
        else if (!moved && Mode == ModeKind.Select)
            Note.Text = "點選  Row " + row.ToString("0.0", CultureInfo.InvariantCulture) + "   Column " + col.ToString("0.0", CultureInfo.InvariantCulture);
        UpdateProbe(at);
    }

    void OnLeave(object sender, MouseEventArgs e)
    {
        if (!Dragging)
            HideProbe();
    }

    void UpdateProbe(Point screen)
    {
        if (PictureData == null)
        {
            HideProbe();
            return;
        }
        (double col, double row) = ImagePoint(screen);
        int ic = (int)Math.Floor(col);
        int ir = (int)Math.Floor(row);
        string where = "Row " + row.ToString("0.0", CultureInfo.InvariantCulture) + "   Column " + col.ToString("0.0", CultureInfo.InvariantCulture);
        string color = "R —  G —  B —";
        if (ic >= 0 && ir >= 0 && ic < PictureData.Width && ir < PictureData.Height)
        {
            int index = (ir * PictureData.Width + ic) * PictureData.Channels;
            if (PictureData.Channels == 1)
            {
                byte gray = PictureData.Pixels[index];
                color = "R " + gray + "  G " + gray + "  B " + gray;
            }
            else
            {
                byte b = PictureData.Pixels[index];
                byte g = PictureData.Pixels[index + 1];
                byte r = PictureData.Pixels[index + 2];
                color = "R " + r + "  G " + g + "  B " + b;
            }
        }
        ProbeText.Text = where + "\n" + color;
        Probe.Visibility = Visibility.Visible;
        Probe.UpdateLayout();
        double left = screen.X + 16;
        double top = screen.Y + 18;
        if (left + Probe.ActualWidth + 8 > Stage.ActualWidth)
            left = screen.X - Probe.ActualWidth - 16;
        if (top + Probe.ActualHeight + 8 > Stage.ActualHeight)
            top = screen.Y - Probe.ActualHeight - 16;
        Canvas.SetLeft(Probe, Math.Max(4, left));
        Canvas.SetTop(Probe, Math.Max(4, top));
        CrossV.X1 = CrossV.X2 = screen.X + 0.5;
        CrossV.Y1 = 0;
        CrossV.Y2 = Stage.ActualHeight;
        CrossH.Y1 = CrossH.Y2 = screen.Y + 0.5;
        CrossH.X1 = 0;
        CrossH.X2 = Stage.ActualWidth;
        CrossV.Visibility = Visibility.Visible;
        CrossH.Visibility = Visibility.Visible;
    }

    void HideProbe()
    {
        Probe.Visibility = Visibility.Collapsed;
        CrossV.Visibility = Visibility.Collapsed;
        CrossH.Visibility = Visibility.Collapsed;
    }

    void ShowZoom(Point a, Point b)
    {
        Canvas.SetLeft(ZoomBox, Math.Min(a.X, b.X));
        Canvas.SetTop(ZoomBox, Math.Min(a.Y, b.Y));
        ZoomBox.Width = Math.Abs(b.X - a.X);
        ZoomBox.Height = Math.Abs(b.Y - a.Y);
        ZoomBox.Visibility = Visibility.Visible;
    }

    static SolidColorBrush Brush(byte r, byte g, byte b) => new(Color.FromRgb(r, g, b));

    static SolidColorBrush Brush(byte a, byte r, byte g, byte b) => new(Color.FromArgb(a, r, g, b));
}
