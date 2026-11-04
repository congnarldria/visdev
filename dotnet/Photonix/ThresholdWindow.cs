using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Vizion;

namespace Photonix;

/// <summary>二值化。獨立視窗，不擋住主視窗。拖統計圖時只改這裡的預覽，放開才寫回程式。</summary>
sealed class ThresholdWindow : Window
{
    public Func<string>? Source { get; set; }
    public Action<string>? ApplySource { get; set; }
    public Action? RequestRun { get; set; }

    readonly Dictionary<string, Button> Tools = new(StringComparer.Ordinal);
    readonly Image Picture = new() { Stretch = Stretch.None, IsHitTestVisible = false };
    readonly HistoBars Bars = new() { Height = 72, Margin = new Thickness(12, 8, 12, 4) };
    readonly TextBlock Note = new() { Foreground = Brush(0x8B, 0x98, 0xAB), Text = "請先在主視窗執行程式，或載入影像。" };
    readonly Grid Stage = new() { Background = Brush(0x10, 0x12, 0x16), ClipToBounds = true };

    ImageBuffer? Image;
    int[] Bins = new int[256];
    int Total;
    ThresholdProgram.State State = ThresholdProgram.Read(null);
    bool Dragging;
    string Drag = "";

    public ThresholdWindow()
    {
        Title = "二值化";
        Width = 920;
        Height = 720;
        MinWidth = 640;
        MinHeight = 480;
        Background = Brush(0x14, 0x17, 0x1C);
        Foreground = Brush(0xD5, 0xDE, 0xEA);
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        var tools = new WrapPanel { Margin = new Thickness(8, 6, 8, 6) };
        foreach ((string key, string label, string tip) in new[]
        {
            ("halcon", "HALCON", "HALCON 區間門檻，含兩端"),
            ("opencv", "OpenCV", "OpenCV，大於門檻"),
            ("inv", "反相", "OpenCV 反相，小於等於門檻"),
            ("otsu", "Otsu", "Otsu，留下亮處"),
            ("dark", "暗部", "Otsu，留下暗處"),
            ("smooth", "平滑", "平滑直方圖後取谷值"),
            ("triangle", "三角", "三角形法"),
            ("local", "局部", "局部平均"),
            ("lgauss", "高斯", "局部高斯"),
        })
        {
            var button = new Button
            {
                Content = label,
                ToolTip = tip,
                Margin = new Thickness(0, 0, 6, 4),
                Padding = new Thickness(8, 3, 8, 3),
                Background = Brush(0x23, 0x28, 0x31),
                Foreground = Brush(0xD5, 0xDE, 0xEA),
                BorderBrush = Brush(0x3A, 0x42, 0x50),
                Tag = key,
            };
            button.Click += (_, _) => UseTool(key);
            Tools[key] = button;
            tools.Children.Add(button);
        }
        var header = new StackPanel { Margin = new Thickness(12, 8, 12, 8) };
        header.Children.Add(new TextBlock { Text = "二值化", FontWeight = FontWeights.Bold });
        header.Children.Add(new TextBlock
        {
            Text = "這個視窗不擋住主視窗。拖統計圖時，區域畫在這裡，放開後主視窗才重跑。",
            Foreground = Brush(0x8B, 0x98, 0xAB),
            TextWrapping = TextWrapping.Wrap,
        });
        Stage.Children.Add(Picture);
        Stage.SizeChanged += (_, _) => PaintPreview();
        Bars.DragBegan += StartDrag;
        Bars.DragMoved += MoveDrag;
        Bars.DragEnded += EndDrag;
        var histo = new Border { Background = Brush(0x16, 0x1A, 0x20), BorderBrush = Brush(0x2C, 0x33, 0x40), BorderThickness = new Thickness(0, 1, 0, 0), Child = Bars };
        var footer = new Border
        {
            Background = Brush(0x1A, 0x1E, 0x25),
            BorderBrush = Brush(0x2C, 0x33, 0x40),
            BorderThickness = new Thickness(0, 1, 0, 0),
            Padding = new Thickness(12, 8, 12, 8),
            Child = Note,
        };
        var root = new DockPanel();
        DockPanel.SetDock(header, Dock.Top);
        DockPanel.SetDock(tools, Dock.Top);
        DockPanel.SetDock(footer, Dock.Bottom);
        DockPanel.SetDock(histo, Dock.Bottom);
        root.Children.Add(header);
        root.Children.Add(tools);
        root.Children.Add(footer);
        root.Children.Add(histo);
        root.Children.Add(Stage);
        Content = root;
    }

    public void ShowImage(ImageBuffer? image, string? source)
    {
        if (Dragging)
            return;
        Image = image;
        State = ThresholdProgram.Read(source);
        RebuildBins();
        MarkTools();
        PaintAll();
    }

    void UseTool(string tool)
    {
        if (Source == null || ApplySource == null)
            return;
        string next = ThresholdProgram.ApplyTool(Source(), tool, State.Level, State.Max);
        ApplySource(next);
        RequestRun?.Invoke();
        ShowImage(Image, next);
    }

    void RebuildBins()
    {
        Bins = new int[256];
        Total = 0;
        if (Image == null)
            return;
        byte[] gray = Image.GrayBytes();
        if (Image.Domain == null)
        {
            foreach (byte value in gray)
                Bins[value]++;
            Total = gray.Length;
            return;
        }
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

    void MarkTools()
    {
        foreach ((string key, Button button) in Tools)
        {
            bool on = key switch
            {
                "halcon" => State.Kind == "range",
                "opencv" => State.Kind == "opencv" && State.LightDark != "binary_inv",
                "inv" => State.Kind == "opencv" && State.LightDark == "binary_inv",
                "otsu" => State.Kind == "otsu" && State.LightDark != "dark",
                "dark" => State.Kind == "otsu" && State.LightDark == "dark",
                "smooth" => State.Method == "smooth_histo",
                "triangle" => State.Method == "triangle",
                "local" => State.Method == "local_mean",
                "lgauss" => State.Method == "local_gauss",
                _ => false,
            };
            button.Background = on ? Brush(0x1D, 0x3D, 0x66) : Brush(0x23, 0x28, 0x31);
            button.BorderBrush = on ? Brush(0x3D, 0x8B, 0xFF) : Brush(0x3A, 0x42, 0x50);
        }
    }

    void PaintAll()
    {
        Bars.Show(Bins, State);
        Note.Text = Stats();
        PaintPreview();
    }

    string Stats()
    {
        if (Image == null)
            return "請先在主視窗執行程式，或載入影像。";
        int area = State.Kind == "range" ? ThresholdProgram.SumBins(Bins, State.Min, State.Max) : ThresholdProgram.SumBins(Bins, State.Level, 255);
        double pct = Total == 0 ? 0 : 100.0 * area / Total;
        string share = area + " px · " + pct.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture) + "%";
        if (State.Kind == "range")
        {
            string hint = State.Literal ? "拖曳兩端，放開後主視窗重跑" : "門檻是算式，只顯示統計";
            return "灰階 " + State.Min + "–" + State.Max + " · 區域 " + share + " · " + hint;
        }
        if (State.Kind == "opencv")
        {
            string rule = State.LightDark == "binary_inv" ? "小於等於" : "大於";
            string hint = State.Literal ? "拖曳門檻，放開後主視窗重跑" : "門檻是算式，只顯示統計";
            return "OpenCV " + rule + " " + State.Level + " · 區域 " + share + " · " + hint;
        }
        return State.Method + " " + State.LightDark + " · 區域 " + share + " · 按了會改程式並重跑";
    }

    void PaintPreview()
    {
        if (Image == null)
        {
            Picture.Source = null;
            return;
        }
        int viewW = Math.Max(2, (int)Math.Round(Stage.ActualWidth));
        int viewH = Math.Max(2, (int)Math.Round(Stage.ActualHeight));
        const double pad = 12;
        double scale = Math.Max(0.05, Math.Min((viewW - pad * 2) / Image.Width, (viewH - pad * 2) / Image.Height));
        double x = (viewW - Image.Width * scale) / 2;
        double y = (viewH - Image.Height * scale) / 2;
        byte[] gray = Image.GrayBytes();
        ImageFrame frame = ImageFrame.Create(Image.Width, Image.Height, gray, PixelLayout.Gray);
        int min = State.Kind == "opencv" && State.LightDark == "binary_inv" ? 0 : State.Kind == "range" ? State.Min : Math.Min(255, State.Level + 1);
        int max = State.Kind == "opencv" && State.LightDark == "binary_inv" ? State.Level : State.Kind == "range" ? State.Max : 255;
        Region region = Operators.Threshold(Image, min, max);
        Overlay[] shapes = region.Rows.Length == 0
            ? []
            : [Overlay.Runs(region.Rows, region.Columns, region.Lengths, 0xFFFF3B30)];
        var map = new ScreenMap(0, 0, viewW, viewH, (0 - y) / scale, (0 - x) / scale, (viewH - y) / scale, (viewW - x) / scale);
        var pixels = new byte[viewW * viewH * 4];
        FrameComposer.Render(frame, map, shapes, pixels, viewW, viewH);
        var source = BitmapSource.Create(viewW, viewH, 96, 96, PixelFormats.Bgra32, null, pixels, viewW * 4);
        source.Freeze();
        Picture.Source = source;
    }

    void StartDrag(int value, double x, double width)
    {
        if (!State.Literal || State.Kind is not ("range" or "opencv"))
            return;
        if (State.Kind == "opencv")
            Drag = "level";
        else
        {
            double minX = State.Min / 256.0 * width;
            double maxX = (State.Max + 1) / 256.0 * width;
            Drag = Math.Abs(x - minX) <= Math.Abs(x - maxX) ? "min" : "max";
        }
        Dragging = true;
        ApplyDrag(value);
    }

    void MoveDrag(int value)
    {
        if (!Dragging)
            return;
        ApplyDrag(value);
    }

    void EndDrag(int value)
    {
        if (!Dragging)
            return;
        Dragging = false;
        ApplyDrag(value);
        if (Source == null || ApplySource == null)
            return;
        string? next = State.Kind == "range"
            ? ThresholdProgram.RewriteRange(Source(), State.Line, State.Min, State.Max) ?? ThresholdProgram.ApplyTool(Source(), "halcon", State.Min, State.Max)
            : ThresholdProgram.RewriteLevel(Source(), State.Line, State.Level);
        if (next == null)
            return;
        ApplySource(next);
        RequestRun?.Invoke();
    }

    void ApplyDrag(int value)
    {
        if (Drag == "level")
            State = State with { Level = value, Max = State.LightDark == "binary_inv" ? value : State.Max, Min = State.LightDark == "binary_inv" ? 0 : value };
        else if (Drag == "min")
            State = State with { Min = Math.Min(value, State.Max) };
        else if (Drag == "max")
            State = State with { Max = Math.Max(value, State.Min) };
        PaintAll();
    }

    static SolidColorBrush Brush(byte r, byte g, byte b) => new(Color.FromRgb(r, g, b));

    sealed class HistoBars : FrameworkElement
    {
        int[] Bins = new int[256];
        ThresholdProgram.State State = ThresholdProgram.Read(null);

        public event Action<int, double, double>? DragBegan;
        public event Action<int>? DragMoved;
        public event Action<int>? DragEnded;

        public void Show(int[] bins, ThresholdProgram.State state)
        {
            Bins = bins;
            State = state;
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
                bool inside = State.Kind == "range" ? i >= State.Min && i <= State.Max : i >= State.Level;
                double bar = Bins[i] / (double)peak * (height - 6);
                drawing.DrawRectangle(inside ? Brush(0x3D, 0x8B, 0xFF) : Brush(0x3A, 0x42, 0x50), null, new Rect(i * slot, height - bar, Math.Max(slot, 1), bar));
            }
            if (State.Kind == "range" && State.Literal)
            {
                Line(drawing, State.Min / 256.0 * width, height, Brush(0xFF, 0xE1, 0x4A));
                Line(drawing, (State.Max + 1) / 256.0 * width, height, Brush(0xFF, 0xE1, 0x4A));
            }
            else
                Line(drawing, (State.Level + 0.5) / 256.0 * width, height, State.Literal ? Brush(0xFF, 0xE1, 0x4A) : Brush(0xFF, 0x8A, 0x1E));
        }

        static void Line(DrawingContext drawing, double x, double height, Brush color) =>
            drawing.DrawLine(new Pen(color, 2), new Point(x, 0), new Point(x, height));

        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            CaptureMouse();
            Point at = e.GetPosition(this);
            DragBegan?.Invoke(ValueAt(at.X), at.X, ActualWidth);
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

        static SolidColorBrush Brush(byte r, byte g, byte b) => new(Color.FromRgb(r, g, b));
    }
}
