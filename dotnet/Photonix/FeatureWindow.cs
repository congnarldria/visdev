using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Vizion;

namespace Photonix;

/// <summary>區域特徵。獨立視窗，不擋住主視窗。先點一個區域或輪廓，再勾選要看的數值。</summary>
sealed class FeatureWindow : Window
{
    public Action? RequestPaint { get; set; }
    public Action<FeatureHit?>? PickChanged { get; set; }

    readonly Image Picture = new() { Stretch = Stretch.Fill, IsHitTestVisible = false };
    readonly TextBlock TitleText = new() { Foreground = Brush(0xFF, 0xE1, 0x4A), TextWrapping = TextWrapping.Wrap, Margin = new Thickness(12, 8, 12, 8), Text = "請點一個區域或輪廓。沒點到之前不顯示數值。" };
    readonly StackPanel Rows = new();
    readonly StackPanel Custom = new();
    readonly Grid Stage = new() { Background = Brush(0x10, 0x12, 0x16), Height = 280, ClipToBounds = true };
    readonly Dictionary<string, CheckBox> Checks = new(StringComparer.Ordinal);
    readonly Dictionary<string, TextBlock> Values = new(StringComparer.Ordinal);
    readonly Dictionary<string, Button> Modes = new(StringComparer.Ordinal);

    ImageBuffer? Image;
    Overlay[] Shapes = [];
    IReadOnlyDictionary<string, object> Vars = new Dictionary<string, object>();
    FeatureHit? Pick;
    Dictionary<string, double> Measured = new(StringComparer.Ordinal);
    string Mode = "select";
    double CamX;
    double CamY;
    double Zoom = 1;
    bool UserView;
    int SeenWidth;
    int SeenHeight;
    Pan? Down;

    public FeatureWindow()
    {
        Title = "區域特徵";
        Width = 880;
        Height = 720;
        MinWidth = 640;
        MinHeight = 480;
        Background = Brush(0x14, 0x17, 0x1C);
        Foreground = Brush(0xD5, 0xDE, 0xEA);
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        var header = new StackPanel { Margin = new Thickness(12, 8, 12, 8) };
        header.Children.Add(new TextBlock { Text = "區域特徵", FontWeight = FontWeights.Bold, FontSize = 14 });
        header.Children.Add(new TextBlock
        {
            Text = "一次只顯示點到的那一個。白色虛線是點到的地方。這個視窗不擋住主視窗。",
            Foreground = Brush(0x8B, 0x98, 0xAB),
            TextWrapping = TextWrapping.Wrap,
        });
        var bar = new WrapPanel { Margin = new Thickness(12, 6, 12, 6) };
        bar.Children.Add(new TextBlock { Text = "圖上可點選、移動、放大", Foreground = Brush(0x8B, 0x98, 0xAB), VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(0, 0, 8, 0) });
        foreach ((string key, string label) in new[] { ("select", "點選"), ("pan", "移動"), ("zoom", "放大") })
        {
            var button = new Button { Content = label, Width = 52, Margin = new Thickness(0, 0, 4, 0), Tag = key };
            button.Click += (_, _) => SetMode(key);
            Modes[key] = button;
            bar.Children.Add(button);
        }
        AddCommand(bar, "符合", (_, _) => Fit(false));
        AddCommand(bar, "1:1", (_, _) => OneToOne());
        AddCommand(bar, "−", (_, _) => ZoomAt(0.8, Stage.ActualWidth / 2, Stage.ActualHeight / 2));
        AddCommand(bar, "+", (_, _) => ZoomAt(1.25, Stage.ActualWidth / 2, Stage.ActualHeight / 2));
        Stage.Children.Add(Picture);
        Stage.MouseWheel += OnWheel;
        Stage.MouseLeftButtonDown += OnDown;
        Stage.MouseMove += OnMove;
        Stage.MouseLeftButtonUp += OnUp;
        BuildChecks(FeaturePick.LoadKeys());
        var list = new ScrollViewer { Content = Rows, VerticalScrollBarVisibility = ScrollBarVisibility.Auto, Padding = new Thickness(12, 8, 12, 16) };
        var customBox = new DockPanel { Margin = new Thickness(12, 8, 12, 16) };
        var customTitle = new TextBlock { Text = "自訂變數", Foreground = Brush(0x9E, 0xB0, 0xC8), FontWeight = FontWeights.SemiBold, Margin = new Thickness(0, 0, 0, 8) };
        DockPanel.SetDock(customTitle, Dock.Top);
        customBox.Children.Add(customTitle);
        customBox.Children.Add(new ScrollViewer { Content = Custom, VerticalScrollBarVisibility = ScrollBarVisibility.Auto });
        var split = new Grid();
        split.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(2, GridUnitType.Star) });
        split.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        Grid.SetColumn(customBox, 1);
        split.Children.Add(list);
        split.Children.Add(customBox);
        var root = new DockPanel();
        DockPanel.SetDock(header, Dock.Top);
        DockPanel.SetDock(bar, Dock.Top);
        DockPanel.SetDock(Stage, Dock.Top);
        DockPanel.SetDock(TitleText, Dock.Top);
        root.Children.Add(header);
        root.Children.Add(bar);
        root.Children.Add(Stage);
        root.Children.Add(TitleText);
        root.Children.Add(split);
        Content = root;
        SetMode("select");
    }

    public bool WantsSubpixel => Checks.TryGetValue("subpixel", out CheckBox? box) && box.IsChecked == true;

    public Overlay[] MainOverlays(double zoom)
    {
        if (!WantsSubpixel || Pick is not FeatureHit hit)
            return [];
        EdgePoint[]? line = FeaturePick.Outline(Vars, hit);
        return line == null ? [] : FeaturePick.Dashed(line, zoom);
    }

    public void Show(ImageBuffer? image, Overlay[] shapes, IReadOnlyDictionary<string, object> vars, FeatureHit? pick)
    {
        Image = image;
        Shapes = shapes ?? [];
        Vars = vars;
        if (!Stage.IsMouseCaptured)
            Pick = pick;
        if (Image == null || Image.Width != SeenWidth || Image.Height != SeenHeight)
        {
            SeenWidth = Image?.Width ?? 0;
            SeenHeight = Image?.Height ?? 0;
            UserView = false;
        }
        RefreshValues();
        RefreshCustom();
        if (!UserView)
            Fit(false);
        else
            Paint();
    }

    void BuildChecks(HashSet<string> chosen)
    {
        AddGroup("區域", FeaturePick.RegionNames.Select(name => ("region:" + name, name, "")));
        AddGroup("輝度", FeaturePick.GrayNames.Select(name => ("gray:" + name, name, FeaturePick.GrayLabels[name])));
        AddGroup("XLD", FeaturePick.XldNames.Select(name => ("xld:" + name, name, "")));
        AddGroup("輪廓", [("subpixel", "次像素 contour", "畫在主視窗")]);
        foreach ((string key, CheckBox box) in Checks)
            box.IsChecked = chosen.Contains(key);
    }

    void AddGroup(string heading, IEnumerable<(string Key, string Name, string Hint)> items)
    {
        Rows.Children.Add(new TextBlock { Text = heading, Foreground = Brush(0x9E, 0xB0, 0xC8), Margin = new Thickness(0, 10, 0, 4) });
        foreach ((string key, string name, string hint) in items)
        {
            var row = new DockPanel { Margin = new Thickness(0, 1, 0, 1) };
            var box = new CheckBox { VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(0, 0, 8, 0), Tag = key };
            box.Checked += (_, _) => OnCheck();
            box.Unchecked += (_, _) => OnCheck();
            var label = new TextBlock { FontFamily = new FontFamily("Consolas"), VerticalAlignment = VerticalAlignment.Center, Text = name };
            if (hint.Length > 0)
                label.Text += "  " + hint;
            var value = new TextBlock { FontFamily = new FontFamily("Consolas"), HorizontalAlignment = HorizontalAlignment.Right, MinWidth = 72 };
            DockPanel.SetDock(box, Dock.Left);
            DockPanel.SetDock(value, Dock.Right);
            row.Children.Add(box);
            row.Children.Add(value);
            row.Children.Add(label);
            Rows.Children.Add(row);
            Checks[key] = box;
            Values[key] = value;
        }
    }

    void OnCheck()
    {
        FeaturePick.SaveKeys(Checks.Where(pair => pair.Value.IsChecked == true).Select(pair => pair.Key));
        RefreshValues();
        RequestPaint?.Invoke();
    }

    void RefreshValues()
    {
        if (Pick is not FeatureHit hit)
        {
            TitleText.Text = "請點一個區域或輪廓。沒點到之前不顯示數值。";
            Measured = new Dictionary<string, double>(StringComparer.Ordinal);
        }
        else
        {
            string kind = hit.Kind == "xld" ? "輪廓" : "區域";
            TitleText.Text = kind + " " + hit.Name + "  #" + (hit.Index + 1) + "（只顯示這一個）";
            Measured = FeaturePick.Measure(Vars, hit, Image);
        }
        foreach ((string key, TextBlock value) in Values)
        {
            if (key == "subpixel" || !Checks.TryGetValue(key, out CheckBox? box) || box.IsChecked != true)
            {
                value.Text = "";
                continue;
            }
            if (Pick is FeatureHit current && current.Kind == "xld" && key.StartsWith("gray:", StringComparison.Ordinal))
            {
                value.Text = "輪廓沒有輝度";
                continue;
            }
            if (key.StartsWith("gray:", StringComparison.Ordinal) && (Image == null || !Measured.ContainsKey(key)))
            {
                value.Text = Image == null ? "無影像" : "";
                continue;
            }
            value.Text = Measured.TryGetValue(key, out double number) ? FeaturePick.Format(number) : "";
        }
    }

    void RefreshCustom()
    {
        Custom.Children.Clear();
        List<VarText> rows = ProgramVars.Describe(Vars);
        if (rows.Count == 0)
        {
            Custom.Children.Add(new TextBlock { Text = "執行程式後，數值會靠右顯示。", Foreground = Brush(0x8B, 0x98, 0xAB) });
            return;
        }
        foreach (VarText item in rows)
        {
            var line = new DockPanel { Margin = new Thickness(0, 2, 0, 2) };
            var name = new TextBlock { Text = item.Name, FontFamily = new FontFamily("Consolas"), Foreground = Brush(0xD5, 0xFB, 0xE6) };
            var text = new TextBlock { Text = item.Short, FontFamily = new FontFamily("Consolas"), HorizontalAlignment = HorizontalAlignment.Right };
            DockPanel.SetDock(text, Dock.Right);
            line.Children.Add(text);
            line.Children.Add(name);
            Custom.Children.Add(line);
        }
    }

    void Paint()
    {
        int viewW = Math.Max(2, (int)Math.Round(Stage.ActualWidth));
        int viewH = Math.Max(2, (int)Math.Round(Stage.ActualHeight));
        if (Image == null)
        {
            Picture.Source = null;
            return;
        }
        var shapes = new List<Overlay>(Shapes);
        if (Pick is FeatureHit hit)
        {
            EdgePoint[]? line = FeaturePick.Outline(Vars, hit);
            if (line != null)
                shapes.AddRange(FeaturePick.Dashed(line, Zoom));
        }
        byte[] pixels = Image.ForDisplay();
        ImageFrame frame = ImageFrame.Create(Image.Width, Image.Height, pixels, Image.Channels == 1 ? PixelLayout.Gray : PixelLayout.Bgr);
        var map = new ScreenMap(0, 0, viewW, viewH, (0 - CamY) / Zoom, (0 - CamX) / Zoom, (viewH - CamY) / Zoom, (viewW - CamX) / Zoom);
        var buffer = new byte[viewW * viewH * 4];
        FrameComposer.Render(frame, map, shapes.ToArray(), buffer, viewW, viewH);
        var source = BitmapSource.Create(viewW, viewH, 96, 96, PixelFormats.Bgra32, null, buffer, viewW * 4);
        source.Freeze();
        Picture.Source = source;
    }

    void Fit(bool markUser)
    {
        if (Image == null || Stage.ActualWidth < 2)
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
        if (Image == null)
            return;
        Stage.CaptureMouse();
        Point at = e.GetPosition(Stage);
        Down = new Pan(at.X, at.Y, CamX, CamY, (at.X - CamX) / Zoom, (at.Y - CamY) / Zoom, Mode);
    }

    void OnMove(object sender, MouseEventArgs e)
    {
        if (Down is not Pan pan || !Stage.IsMouseCaptured || pan.Mode != "pan")
            return;
        Point at = e.GetPosition(Stage);
        CamX = pan.CamX + at.X - pan.X;
        CamY = pan.CamY + at.Y - pan.Y;
        UserView = true;
        Paint();
    }

    void OnUp(object sender, MouseButtonEventArgs e)
    {
        if (!Stage.IsMouseCaptured)
            return;
        Stage.ReleaseMouseCapture();
        Pan? start = Down;
        Down = null;
        if (start is not Pan pan || Image == null)
            return;
        Point at = e.GetPosition(Stage);
        double moved = Math.Sqrt((at.X - pan.X) * (at.X - pan.X) + (at.Y - pan.Y) * (at.Y - pan.Y));
        double column = (at.X - CamX) / Zoom;
        double row = (at.Y - CamY) / Zoom;
        if (pan.Mode == "zoom")
        {
            if (moved > 4)
            {
                double left = Math.Min(pan.Column, column);
                double top = Math.Min(pan.Row, row);
                double width = Math.Max(1, Math.Abs(column - pan.Column));
                double height = Math.Max(1, Math.Abs(row - pan.Row));
                Zoom = Math.Min(48, Math.Max(0.05, Math.Min((Stage.ActualWidth - 16) / width, (Stage.ActualHeight - 16) / height)));
                CamX = (Stage.ActualWidth - width * Zoom) / 2 - left * Zoom;
                CamY = (Stage.ActualHeight - height * Zoom) / 2 - top * Zoom;
                UserView = true;
            }
            else
                ZoomAt(1.4, at.X, at.Y);
            Paint();
            return;
        }
        if (pan.Mode == "pan" || moved > 4)
            return;
        PickChanged?.Invoke(FeaturePick.Find(Vars, pan.Column, pan.Row, Zoom));
    }

    void SetMode(string mode)
    {
        Mode = mode is "pan" or "zoom" ? mode : "select";
        foreach ((string key, Button button) in Modes)
        {
            bool on = key == Mode;
            button.Background = on ? Brush(0x1D, 0x4E, 0x86) : Brush(0x23, 0x28, 0x31);
            button.BorderBrush = on ? Brush(0x3D, 0x8B, 0xFF) : Brush(0x3A, 0x42, 0x50);
            button.Foreground = Brush(0xD5, 0xDE, 0xEA);
        }
        Stage.Cursor = Mode == "pan" ? Cursors.Hand : Mode == "zoom" ? Cursors.Cross : Cursors.Arrow;
    }

    static void AddCommand(Panel panel, string label, RoutedEventHandler click)
    {
        var button = new Button { Content = label, Margin = new Thickness(4, 0, 0, 0) };
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
}
