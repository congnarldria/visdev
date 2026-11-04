using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Vizion;

namespace Photonix;

/// <summary>次像素二值化。獨立視窗，不擋住主視窗。拖統計圖時只改這裡，放開才寫回程式。</summary>
sealed class SubpixWindow : Window
{
    public Func<string>? Source { get; set; }
    public Action<string>? ApplySource { get; set; }
    public Action? RequestRun { get; set; }

    readonly Image Picture = new() { Stretch = Stretch.Fill, IsHitTestVisible = false };
    readonly HistoBars Bars = new() { Height = 72, Margin = new Thickness(12, 8, 12, 4), Cursor = Cursors.SizeWE };
    readonly TextBlock Note = new() { Foreground = Brush(0x8B, 0x98, 0xAB), Text = "區域與輪廓會同時留在主視窗。", TextWrapping = TextWrapping.Wrap };
    readonly TextBlock Probe = new()
    {
        Foreground = Brush(0xF4, 0xF7, 0xFB),
        Background = Brush(0x10, 0x12, 0x16),
        Padding = new Thickness(8, 4, 8, 4),
        FontFamily = new FontFamily("Consolas"),
        Visibility = Visibility.Collapsed,
        IsHitTestVisible = false,
    };
    readonly System.Windows.Shapes.Rectangle Band = new()
    {
        Stroke = Brush(0xFF, 0xE1, 0x4A),
        StrokeThickness = 1,
        Visibility = Visibility.Collapsed,
        IsHitTestVisible = false,
    };
    readonly TextBox LevelBox = new() { Width = 72, Text = "128", Background = Brush(0x10, 0x12, 0x16), Foreground = Brush(0xD5, 0xDE, 0xEA), BorderBrush = Brush(0x3A, 0x42, 0x50) };
    readonly TextBox WidthBox = new() { Width = 48, Text = "2" };
    readonly TextBlock ZoomText = new() { Foreground = Brush(0x8B, 0x98, 0xAB), Margin = new Thickness(8, 0, 8, 0) };
    readonly Grid Stage = new() { Background = Brush(0x10, 0x12, 0x16), ClipToBounds = true };
    readonly Dictionary<string, Button> Modes = new(StringComparer.Ordinal);

    ImageBuffer? Image;
    SubpixFigure? Figure;
    SubpixProgram.State Names = SubpixProgram.Read(null);
    int[] Bins = new int[256];
    int Total;
    int Level = 128;
    int LineWidth = 2;
    bool Dragging;
    bool SuppressLevel;
    string Mode = "pan";
    double CamX;
    double CamY = 0;
    double Zoom = 1;
    bool UserView;
    int SeenWidth;
    int SeenHeight;
    Pan? Down;
    int? Picked;

    public SubpixWindow()
    {
        Title = "次像素二值化";
        Width = 920;
        Height = 720;
        MinWidth = 640;
        MinHeight = 480;
        Background = Brush(0x14, 0x17, 0x1C);
        Foreground = Brush(0xD5, 0xDE, 0xEA);
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        var header = new StackPanel { Margin = new Thickness(12, 8, 12, 8) };
        header.Children.Add(new TextBlock { Text = "次像素二值化", FontWeight = FontWeights.Bold });
        header.Children.Add(new TextBlock
        {
            Text = "八連通自動拆成一個一個物件。點到的那一個才畫白色虛線。這個視窗不擋住主視窗。",
            Foreground = Brush(0x8B, 0x98, 0xAB),
            TextWrapping = TextWrapping.Wrap,
        });
        var bar = new WrapPanel { Margin = new Thickness(8, 4, 8, 4), Background = Brush(0x16, 0x1A, 0x20) };
        foreach ((string key, string tip) in new[] { ("select", "箭頭點選區域或輪廓"), ("pan", "手抓移動畫面"), ("zoom", "拉一個範圍放大。點一下也會放大") })
        {
            var button = new Button
            {
                Content = key == "select" ? "點選" : key == "pan" ? "手抓" : "放大",
                ToolTip = tip,
                Width = 52,
                Margin = new Thickness(0, 0, 2, 0),
                Tag = key,
            };
            button.Click += (_, _) => SetMode(key);
            Modes[key] = button;
            bar.Children.Add(button);
        }
        AddCommand(bar, "符合", "符合視窗", (_, _) => Fit(false));
        AddCommand(bar, "1:1", "原始像素", (_, _) => OneToOne());
        AddCommand(bar, "−", "縮小", (_, _) => ZoomAt(0.8, Stage.ActualWidth / 2, Stage.ActualHeight / 2));
        AddCommand(bar, "+", "放大", (_, _) => ZoomAt(1.25, Stage.ActualWidth / 2, Stage.ActualHeight / 2));
        bar.Children.Add(new TextBlock { Text = "線寬", Foreground = Brush(0x8B, 0x98, 0xAB), VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(8, 0, 4, 0) });
        WidthBox.LostFocus += (_, _) => ApplyWidth();
        WidthBox.KeyDown += (_, e) =>
        {
            if (e.Key == Key.Enter)
                ApplyWidth();
        };
        bar.Children.Add(WidthBox);
        bar.Children.Add(ZoomText);
        bar.Children.Add(new TextBlock { Text = "滾輪縮放。拖統計圖看結果，放開後主視窗重跑。", Foreground = Brush(0x8B, 0x98, 0xAB), VerticalAlignment = VerticalAlignment.Center });
        var canvas = new Canvas { IsHitTestVisible = false };
        canvas.Children.Add(Band);
        canvas.Children.Add(Probe);
        Stage.Children.Add(Picture);
        Stage.Children.Add(canvas);
        Stage.SizeChanged += (_, _) => Paint();
        Stage.MouseWheel += OnWheel;
        Stage.MouseLeftButtonDown += OnDown;
        Stage.MouseMove += OnMove;
        Stage.MouseLeftButtonUp += OnUp;
        Stage.MouseLeave += (_, _) => Probe.Visibility = Visibility.Collapsed;
        Bars.DragBegan += value =>
        {
            Dragging = true;
            SetLevel(value, true);
        };
        Bars.DragMoved += value =>
        {
            if (Dragging)
                SetLevel(value, true);
        };
        Bars.DragEnded += value =>
        {
            if (!Dragging)
                return;
            Dragging = false;
            SetLevel(value, true);
            Commit();
        };
        var histo = new Border { Background = Brush(0x16, 0x1A, 0x20), BorderBrush = Brush(0x2C, 0x33, 0x40), BorderThickness = new Thickness(0, 1, 0, 0), Child = Bars };
        var footer = new WrapPanel { Background = Brush(0x1A, 0x1E, 0x25), Margin = new Thickness(0) };
        var foot = new Border
        {
            Background = Brush(0x1A, 0x1E, 0x25),
            BorderBrush = Brush(0x2C, 0x33, 0x40),
            BorderThickness = new Thickness(0, 1, 0, 0),
            Padding = new Thickness(12, 8, 12, 8),
            Child = footer,
        };
        footer.Children.Add(new TextBlock { Text = "門檻", VerticalAlignment = VerticalAlignment.Center, Foreground = Brush(0x8B, 0x98, 0xAB), Margin = new Thickness(0, 0, 4, 0) });
        LevelBox.LostFocus += (_, _) => ReadLevel(false);
        LevelBox.KeyDown += (_, e) =>
        {
            if (e.Key == Key.Enter)
                ReadLevel(false);
        };
        footer.Children.Add(LevelBox);
        var apply = new Button { Content = "套用到程式", Margin = new Thickness(8, 0, 8, 0) };
        apply.Click += (_, _) =>
        {
            ReadLevel(false);
            Commit();
        };
        footer.Children.Add(apply);
        footer.Children.Add(Note);
        var root = new DockPanel();
        DockPanel.SetDock(header, Dock.Top);
        DockPanel.SetDock(bar, Dock.Top);
        DockPanel.SetDock(foot, Dock.Bottom);
        DockPanel.SetDock(histo, Dock.Bottom);
        root.Children.Add(header);
        root.Children.Add(bar);
        root.Children.Add(foot);
        root.Children.Add(histo);
        root.Children.Add(Stage);
        Content = root;
        SetMode("pan");
    }

    public void ShowImage(ImageBuffer? image, string? source)
    {
        if (Dragging)
            return;
        Image = image;
        if (!LevelBox.IsKeyboardFocusWithin)
        {
            Names = SubpixProgram.Read(source);
            SetLevel(Names.Level, false);
        }
        Rebuild();
    }

    void Commit()
    {
        if (Source == null || ApplySource == null)
            return;
        string next = SubpixProgram.Apply(Source(), Level);
        ApplySource(next);
        RequestRun?.Invoke();
        Note.Text = "已套用。主視窗同時保留區域和輪廓。";
    }

    void SetLevel(int value, bool preview)
    {
        Level = Math.Clamp(value, 0, 255);
        SuppressLevel = true;
        LevelBox.Text = Level.ToString(CultureInfo.InvariantCulture);
        SuppressLevel = false;
        if (preview)
            Rebuild();
    }

    void ReadLevel(bool commit)
    {
        if (SuppressLevel)
            return;
        if (!int.TryParse(LevelBox.Text.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int value))
            value = Level;
        SetLevel(value, true);
        if (commit)
            Commit();
    }

    void ApplyWidth()
    {
        if (!int.TryParse(WidthBox.Text.Trim(), out int value))
            value = LineWidth;
        LineWidth = Math.Clamp(value, 1, 10);
        WidthBox.Text = LineWidth.ToString(CultureInfo.InvariantCulture);
        Paint();
    }

    void Rebuild()
    {
        Bins = new int[256];
        Total = 0;
        Figure = null;
        if (Image != null)
        {
            byte[] gray = Image.GrayBytes();
            if (Image.Domain == null)
            {
                foreach (byte value in gray)
                    Bins[value]++;
                Total = gray.Length;
            }
            else
            {
                Region domain = Image.Domain;
                for (int i = 0; i < domain.Rows.Length; i++)
                {
                    int offset = domain.Rows[i] * Image.Width + domain.Columns[i];
                    int length = domain.Lengths[i];
                    for (int k = 0; k < length; k++)
                    {
                        Bins[gray[offset + k]]++;
                        Total++;
                    }
                }
            }
            try
            {
                Figure = Operators.ThresholdSubPix(Image, Level);
            }
            catch (Exception ex)
            {
                Note.Text = ex.Message;
                Bars.Show(Bins, Level);
                Paint();
                return;
            }
        }
        Bars.Show(Bins, Level);
        Note.Text = Stats();
        if (Image == null || Image.Width != SeenWidth || Image.Height != SeenHeight)
        {
            SeenWidth = Image?.Width ?? 0;
            SeenHeight = Image?.Height ?? 0;
            UserView = false;
            Picked = null;
        }
        if (!UserView)
            Fit(false);
        else
            Paint();
    }

    string Stats()
    {
        if (Image == null)
            return "請先在主視窗執行一次程式。";
        int area = ThresholdProgram.SumBins(Bins, Level, 255);
        double pct = Total == 0 ? 0 : 100.0 * area / Total;
        int count = Figure?.Objects.Length ?? 0;
        int contours = Figure?.Contours.Length ?? 0;
        return "區域 " + area + " px · " + pct.ToString("0.0", CultureInfo.InvariantCulture) + "% · 物件 " + count + " 個 · 輪廓 " + contours + " 條 · 門檻 " + Level + "。八連通分開，點一個只框那一個。";
    }

    void Paint()
    {
        int viewW = Math.Max(2, (int)Math.Round(Stage.ActualWidth));
        int viewH = Math.Max(2, (int)Math.Round(Stage.ActualHeight));
        ZoomText.Text = Image == null ? "" : Math.Round(Zoom * 100).ToString(CultureInfo.InvariantCulture) + "%";
        if (Image == null)
        {
            Picture.Source = null;
            return;
        }
        byte[] gray = Image.GrayBytes();
        ImageFrame frame = ImageFrame.Create(Image.Width, Image.Height, gray, PixelLayout.Gray);
        var shapes = new List<Overlay>();
        if (Figure != null)
            shapes.AddRange(SubpixFigure.Overlays(Figure, LineWidth));
        if (Picked is int picked && Figure != null)
        {
            double dash = 7 / Math.Max(Zoom, 0.05);
            double gap = 5 / Math.Max(Zoom, 0.05);
            for (int i = 0; i < Figure.Contours.Length; i++)
            {
                if (Figure.Groups[i] != picked)
                    continue;
                AddDashed(shapes, Figure.Contours[i], dash, gap, 0xFF000000, 4);
                AddDashed(shapes, Figure.Contours[i], dash, gap, 0xFFFFFFFF, 2);
            }
        }
        var map = new ScreenMap(0, 0, viewW, viewH, (0 - CamY) / Zoom, (0 - CamX) / Zoom, (viewH - CamY) / Zoom, (viewW - CamX) / Zoom);
        var pixels = new byte[viewW * viewH * 4];
        FrameComposer.Render(frame, map, shapes.ToArray(), pixels, viewW, viewH);
        var source = BitmapSource.Create(viewW, viewH, 96, 96, PixelFormats.Bgra32, null, pixels, viewW * 4);
        source.Freeze();
        Picture.Source = source;
    }

    static void AddDashed(List<Overlay> overlays, EdgePoint[] chain, double dash, double gap, uint color, int stroke)
    {
        if (chain.Length < 2 || dash <= 0)
            return;
        bool on = true;
        double drawn = 0;
        var rows = new List<double> { chain[0].Row };
        var columns = new List<double> { chain[0].Column };
        void Flush()
        {
            if (rows.Count >= 2)
                overlays.Add(Overlay.Polyline(rows.ToArray(), columns.ToArray(), color).WithStroke(stroke));
            rows.Clear();
            columns.Clear();
        }
        for (int i = 1; i < chain.Length; i++)
        {
            double row0 = chain[i - 1].Row;
            double column0 = chain[i - 1].Column;
            double row1 = chain[i].Row;
            double column1 = chain[i].Column;
            double dr = row1 - row0;
            double dc = column1 - column0;
            double length = Math.Sqrt(dr * dr + dc * dc);
            if (length < 1e-9)
                continue;
            double walked = 0;
            while (walked < length - 1e-9)
            {
                double room = (on ? dash : gap) - drawn;
                double step = Math.Min(room, length - walked);
                walked += step;
                drawn += step;
                double t = walked / length;
                double row = row0 + dr * t;
                double column = column0 + dc * t;
                if (on)
                {
                    rows.Add(row);
                    columns.Add(column);
                }
                if (drawn >= (on ? dash : gap) - 1e-9)
                {
                    if (on)
                        Flush();
                    on = !on;
                    drawn = 0;
                    if (on)
                    {
                        rows.Add(row);
                        columns.Add(column);
                    }
                }
            }
        }
        if (on)
            Flush();
    }

    void Fit(bool markUser)
    {
        if (Image == null || Stage.ActualWidth < 2 || Stage.ActualHeight < 2)
        {
            UserView = false;
            Paint();
            return;
        }
        const double pad = 12;
        Zoom = Math.Max(0.05, Math.Min((Stage.ActualWidth - pad * 2) / Image.Width, (Stage.ActualHeight - pad * 2) / Image.Height));
        CamX = (Stage.ActualWidth - Image.Width * Zoom) / 2;
        CamY = (Stage.ActualHeight - Image.Height * Zoom) / 2;
        UserView = markUser;
        Paint();
    }

    void OneToOne()
    {
        if (Image == null)
            return;
        Zoom = 1;
        CamX = (Stage.ActualWidth - Image.Width) / 2;
        CamY = (Stage.ActualHeight - Image.Height) / 2;
        UserView = true;
        Paint();
    }

    void ZoomAt(double factor, double sx, double sy)
    {
        if (Image == null)
            return;
        double next = Math.Min(48, Math.Max(0.05, Zoom * factor));
        double k = next / (Zoom == 0 ? 1 : Zoom);
        CamX = sx - (sx - CamX) * k;
        CamY = sy - (sy - CamY) * k;
        Zoom = next;
        UserView = true;
        Paint();
    }

    (double X, double Y, double Column, double Row) ImagePoint(Point at) => (at.X, at.Y, (at.X - CamX) / Zoom, (at.Y - CamY) / Zoom);

    void OnWheel(object sender, MouseWheelEventArgs e)
    {
        if (Image == null)
            return;
        Point at = e.GetPosition(Stage);
        ZoomAt(e.Delta > 0 ? 1.12 : 1 / 1.12, at.X, at.Y);
        e.Handled = true;
    }

    void OnDown(object sender, MouseButtonEventArgs e)
    {
        if (Image == null || e.ChangedButton != MouseButton.Left)
            return;
        Stage.CaptureMouse();
        var at = ImagePoint(e.GetPosition(Stage));
        Down = new Pan(at.X, at.Y, CamX, CamY, at.Column, at.Row, Mode);
    }

    void OnMove(object sender, MouseEventArgs e)
    {
        if (Image == null)
            return;
        var at = ImagePoint(e.GetPosition(Stage));
        if (Down is Pan pan && pan.Mode == "pan" && Stage.IsMouseCaptured)
        {
            CamX = pan.CamX + at.X - pan.X;
            CamY = pan.CamY + at.Y - pan.Y;
            UserView = true;
            Paint();
        }
        else if (Down is Pan box && box.Mode == "zoom" && Stage.IsMouseCaptured)
        {
            Band.Visibility = Visibility.Visible;
            Canvas.SetLeft(Band, Math.Min(box.X, at.X));
            Canvas.SetTop(Band, Math.Min(box.Y, at.Y));
            Band.Width = Math.Abs(at.X - box.X);
            Band.Height = Math.Abs(at.Y - box.Y);
        }
        ShowProbe(at.X, at.Y, at.Column, at.Row);
    }

    void OnUp(object sender, MouseButtonEventArgs e)
    {
        if (!Stage.IsMouseCaptured)
            return;
        Stage.ReleaseMouseCapture();
        Pan? start = Down;
        Down = null;
        Band.Visibility = Visibility.Collapsed;
        if (start is not Pan pan || Image == null)
            return;
        var at = ImagePoint(e.GetPosition(Stage));
        double moved = Math.Sqrt((at.X - pan.X) * (at.X - pan.X) + (at.Y - pan.Y) * (at.Y - pan.Y));
        if (pan.Mode != "zoom" && moved <= 4)
            PickAt(pan.Column, pan.Row);
        if (pan.Mode != "zoom")
            return;
        if (moved > 4)
        {
            double left = Math.Min(pan.Column, at.Column);
            double top = Math.Min(pan.Row, at.Row);
            double width = Math.Max(1, Math.Abs(at.Column - pan.Column));
            double height = Math.Max(1, Math.Abs(at.Row - pan.Row));
            Zoom = Math.Min(48, Math.Max(0.05, Math.Min((Stage.ActualWidth - 16) / width, (Stage.ActualHeight - 16) / height)));
            CamX = (Stage.ActualWidth - width * Zoom) / 2 - left * Zoom;
            CamY = (Stage.ActualHeight - height * Zoom) / 2 - top * Zoom;
            UserView = true;
        }
        else
            ZoomAt(1.4, at.X, at.Y);
        Paint();
    }

    void PickAt(double column, double row)
    {
        if (Figure == null)
            return;
        int? contour = null;
        double best = 12 / Math.Max(Zoom, 0.05);
        for (int i = 0; i < Figure.Contours.Length; i++)
        {
            double distance = LineDistance(column, row, Figure.Contours[i]);
            if (distance <= best)
            {
                best = distance;
                contour = Figure.Groups[i];
            }
        }
        if (contour is int group)
        {
            Picked = group;
            Note.Text = "已點選輪廓 " + Names.Contour + " #" + (group + 1) + "。白色虛線只框這一個物件。";
            Paint();
            return;
        }
        int col = (int)Math.Floor(column);
        int pixelRow = (int)Math.Floor(row);
        int? region = null;
        int area = int.MaxValue;
        for (int i = 0; i < Figure.Objects.Length; i++)
        {
            if (!Contains(Figure.Objects[i], col, pixelRow))
                continue;
            int size = Figure.Objects[i].Area;
            if (size < area)
            {
                area = size;
                region = i;
            }
        }
        if (region is int index)
        {
            Picked = index;
            Note.Text = "已點選區域 #" + (index + 1) + "。白色虛線只框這一個物件。";
            Paint();
            return;
        }
        Picked = null;
        Note.Text = "沒有點到。點一條輪廓或區域裡面，白色虛線只框那一個物件。";
        Paint();
    }

    static double LineDistance(double column, double row, EdgePoint[] line)
    {
        double best = double.PositiveInfinity;
        for (int i = 1; i < line.Length; i++)
        {
            double ax = line[i - 1].Column;
            double ay = line[i - 1].Row;
            double bx = line[i].Column;
            double by = line[i].Row;
            double dx = bx - ax;
            double dy = by - ay;
            double len2 = dx * dx + dy * dy;
            if (len2 < 1e-12)
                len2 = 1;
            double t = ((column - ax) * dx + (row - ay) * dy) / len2;
            if (t < 0)
                t = 0;
            else if (t > 1)
                t = 1;
            double px = ax + t * dx;
            double py = ay + t * dy;
            double dist = (column - px) * (column - px) + (row - py) * (row - py);
            if (dist < best)
                best = dist;
        }
        return Math.Sqrt(best);
    }

    static bool Contains(Region region, int column, int row)
    {
        for (int i = 0; i < region.Rows.Length; i++)
        {
            if (region.Rows[i] != row)
                continue;
            int start = region.Columns[i];
            if (column >= start && column < start + region.Lengths[i])
                return true;
        }
        return false;
    }

    void ShowProbe(double x, double y, double column, double row)
    {
        if (Image == null)
            return;
        int col = (int)Math.Floor(column);
        int pixelRow = (int)Math.Floor(row);
        string color = "R —  G —  B —";
        if (col >= 0 && pixelRow >= 0 && col < Image.Width && pixelRow < Image.Height)
        {
            int offset = (pixelRow * Image.Width + col) * Image.Channels;
            if (Image.Channels == 1)
            {
                byte gray = Image.Pixels[offset];
                color = "R " + gray + "  G " + gray + "  B " + gray;
            }
            else
            {
                byte b = Image.Pixels[offset];
                byte g = Image.Pixels[offset + 1];
                byte r = Image.Pixels[offset + 2];
                color = "R " + r + "  G " + g + "  B " + b;
            }
        }
        Probe.Text = "Row " + row.ToString("0.0", CultureInfo.InvariantCulture) + "   Column " + column.ToString("0.0", CultureInfo.InvariantCulture) + "\n" + color;
        Probe.Visibility = Visibility.Visible;
        Probe.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
        double left = x + 16;
        double top = y + 18;
        if (left + Probe.DesiredSize.Width + 8 > Stage.ActualWidth)
            left = x - Probe.DesiredSize.Width - 16;
        if (top + Probe.DesiredSize.Height + 8 > Stage.ActualHeight)
            top = y - Probe.DesiredSize.Height - 16;
        Canvas.SetLeft(Probe, Math.Max(4, left));
        Canvas.SetTop(Probe, Math.Max(4, top));
    }

    void SetMode(string mode)
    {
        Mode = mode is "select" or "zoom" ? mode : "pan";
        foreach ((string key, Button button) in Modes)
        {
            bool on = key == Mode;
            button.Background = on ? Brush(0x1D, 0x4E, 0x86) : Brush(0x23, 0x28, 0x31);
            button.BorderBrush = on ? Brush(0x3D, 0x8B, 0xFF) : Brush(0x3A, 0x42, 0x50);
            button.Foreground = Brush(0xD5, 0xDE, 0xEA);
        }
        Stage.Cursor = Mode == "pan" ? Cursors.Hand : Mode == "zoom" ? Cursors.Cross : Cursors.Arrow;
    }

    static void AddCommand(Panel panel, string label, string tip, RoutedEventHandler click)
    {
        var button = new Button { Content = label, ToolTip = tip, Margin = new Thickness(4, 0, 0, 0) };
        button.Click += click;
        panel.Children.Add(button);
    }

    static SolidColorBrush Brush(byte r, byte g, byte b)
    {
        var brush = new SolidColorBrush(Color.FromRgb(r, g, b));
        brush.Freeze();
        return brush;
    }

    readonly record struct Pan(double X, double Y, double CamX, double CamY, double Column, double Row, string Mode);

    sealed class HistoBars : FrameworkElement
    {
        int[] Bins = new int[256];
        int Level = 128;

        public event Action<int>? DragBegan;
        public event Action<int>? DragMoved;
        public event Action<int>? DragEnded;

        public void Show(int[] bins, int level)
        {
            Bins = bins;
            Level = level;
            InvalidateVisual();
        }

        protected override void OnRender(DrawingContext drawing)
        {
            double width = ActualWidth;
            double height = ActualHeight;
            if (width < 2 || height < 2)
                return;
            drawing.DrawRectangle(Brush(0x10, 0x12, 0x16), null, new Rect(0, 0, width, height));
            int peak = 1;
            foreach (int value in Bins)
                peak = Math.Max(peak, value);
            double slot = width / 256;
            for (int i = 0; i < 256; i++)
            {
                double bar = Bins[i] / (double)peak * (height - 6);
                drawing.DrawRectangle(i >= Level ? Brush(0x3D, 0x8B, 0xFF) : Brush(0x3A, 0x42, 0x50), null, new Rect(i * slot, height - bar, Math.Max(slot, 1), bar));
            }
            double x = (Level + 0.5) / 256.0 * width;
            drawing.DrawLine(new Pen(Brush(0xFF, 0xE1, 0x4A), 2), new Point(x, 0), new Point(x, height));
        }

        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            if (Bins.Length == 0 || Bins.All(value => value == 0))
                return;
            CaptureMouse();
            DragBegan?.Invoke(ValueAt(e.GetPosition(this).X));
            e.Handled = true;
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            if (!IsMouseCaptured)
                return;
            DragMoved?.Invoke(ValueAt(e.GetPosition(this).X));
        }

        protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e)
        {
            if (!IsMouseCaptured)
                return;
            ReleaseMouseCapture();
            DragEnded?.Invoke(ValueAt(e.GetPosition(this).X));
        }

        int ValueAt(double x)
        {
            double width = Math.Max(1, ActualWidth);
            double clamped = Math.Min(Math.Max(x, 0), width);
            return Math.Min(255, Math.Max(0, (int)Math.Floor(clamped / width * 256)));
        }

        static SolidColorBrush Brush(byte r, byte g, byte b)
        {
            var brush = new SolidColorBrush(Color.FromRgb(r, g, b));
            brush.Freeze();
            return brush;
        }
    }
}
