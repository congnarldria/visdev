using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using Vizion;

namespace Vizion.Ide;

public partial class MainWindow : Window
{
    readonly OperatorItem[] Catalog =
    {
        new("read_image", "檔案", "讀入一張 PNG。", new[] { "路徑" }),
        new("threshold", "區域", "灰階落在兩端之間的像素收成區域。", new[] { "最小灰階", "最大灰階" }),
        new("connection", "區域", "把區域拆成連通物件。", new[] { "距離" }),
        new("mean_image", "濾波", "平均濾波。", new[] { "遮罩寬", "遮罩高" }),
        new("gauss_image", "濾波", "高斯濾波。", new[] { "大小" }),
        new("edges_sub_pix", "邊緣", "次像素邊緣。", new[] { "Alpha", "Low", "High" }),
        new("find_shape", "比對", "用邊緣方向找模板。", new[] { "Contrast", "MinContrast", "MinScore" }),
        new("fit_line", "量測", "用點擬合直線，飛點會再擬一次。", new[] { "點" }),
        new("fit_circle", "量測", "用點擬合圓，飛點會再擬一次。", new[] { "點" }),
    };

    ImageBuffer? PictureData;
    double Zoom = 1;
    double CamX;
    double CamY;
    bool PendingFit;
    bool Dragging;
    Point DragStart;
    double DragCamX;
    double DragCamY;
    double DragCol;
    double DragRow;
    GfxMode Mode = GfxMode.Select;
    readonly Brush ModeOff = new SolidColorBrush(Color.FromRgb(0x21, 0x26, 0x2D));

    public MainWindow()
    {
        InitializeComponent();
        OperatorList.ItemsSource = Catalog;
        ShowTab("ops");
        ApplyMode(GfxMode.Select);
        Log("VisDev 已開啟。版面對齊網頁版，底層是 Vizion.Operators。");
        if (NativeCore.IsLoaded)
            Log("已載入 C 核心：" + NativeCore.LoadDetail);
        else
            Log("沒有載入 C 核心，運算會用 C#：" + NativeCore.LoadDetail);
    }

    void LoadImage_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Filter = "PNG (*.png)|*.png|所有檔案 (*.*)|*.*",
            Title = "載入影像",
        };
        if (dialog.ShowDialog(this) != true)
            return;
        ShowPath(dialog.FileName);
    }

    void ShowPath(string path)
    {
        try
        {
            PictureData = Operators.ReadImage(path);
            Picture.Source = ToBitmap(PictureData);
            Picture.Visibility = Visibility.Visible;
            EmptyHint.Visibility = Visibility.Collapsed;
            Fit();
            Log($"載入 {path}，{PictureData.Width}×{PictureData.Height}，{PictureData.Channels} 通道。");
        }
        catch (Exception ex)
        {
            Log(ex.Message);
        }
    }

    void Fit_Click(object sender, RoutedEventArgs e) => Fit();

    void OneToOne_Click(object sender, RoutedEventArgs e)
    {
        if (PictureData == null)
            return;
        ApplyView(1, (Stage.ActualWidth - PictureData.Width) / 2, (Stage.ActualHeight - PictureData.Height) / 2);
    }

    void ZoomIn_Click(object sender, RoutedEventArgs e) => ZoomAtCenter(1.25);

    void ZoomOut_Click(object sender, RoutedEventArgs e) => ZoomAtCenter(0.8);

    void Mode_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not Button button)
            return;
        ApplyMode((button.Tag as string) switch
        {
            "pan" => GfxMode.Pan,
            "zoom" => GfxMode.Zoom,
            _ => GfxMode.Select,
        });
    }

    void Stage_DragOver(object sender, DragEventArgs e)
    {
        e.Effects = e.Data.GetDataPresent(DataFormats.FileDrop) ? DragDropEffects.Copy : DragDropEffects.None;
        e.Handled = true;
    }

    void Stage_Drop(object sender, DragEventArgs e)
    {
        if (e.Data.GetData(DataFormats.FileDrop) is not string[] files || files.Length == 0)
            return;
        ShowPath(files[0]);
        e.Handled = true;
    }

    void Stage_MouseWheel(object sender, MouseWheelEventArgs e)
    {
        if (PictureData == null)
            return;
        Point at = e.GetPosition(Stage);
        ZoomAt(at.X, at.Y, e.Delta > 0 ? 1.12 : 1 / 1.12);
        UpdateProbe(at);
        e.Handled = true;
    }

    void Stage_MouseDown(object sender, MouseButtonEventArgs e)
    {
        if (PictureData == null)
            return;
        Stage.CaptureMouse();
        Dragging = true;
        DragStart = e.GetPosition(Stage);
        DragCamX = CamX;
        DragCamY = CamY;
        (DragCol, DragRow) = ImagePoint(DragStart);
        if (Mode == GfxMode.Zoom)
            ShowZoomRect(DragStart, DragStart);
    }

    void Stage_MouseMove(object sender, MouseEventArgs e)
    {
        Point at = e.GetPosition(Stage);
        if (Dragging && PictureData != null && Mode == GfxMode.Pan)
            ApplyView(Zoom, DragCamX + at.X - DragStart.X, DragCamY + at.Y - DragStart.Y);
        else if (Dragging && Mode == GfxMode.Zoom)
            ShowZoomRect(DragStart, at);
        UpdateProbe(at);
    }

    void Stage_MouseUp(object sender, MouseButtonEventArgs e)
    {
        if (!Dragging)
            return;
        Dragging = false;
        Stage.ReleaseMouseCapture();
        ZoomRect.Visibility = Visibility.Collapsed;
        if (PictureData == null)
            return;
        Point at = e.GetPosition(Stage);
        (double col, double row) = ImagePoint(at);
        double dx = at.X - DragStart.X;
        double dy = at.Y - DragStart.Y;
        bool moved = dx * dx + dy * dy > 16;
        if (Mode == GfxMode.Zoom)
        {
            if (moved)
                ZoomToRect(DragCol, DragRow, col, row);
            else
                ZoomAt(at.X, at.Y, 1.4);
        }
        else if (!moved && Mode == GfxMode.Select)
            StatusLine.Text = $"點選  Row {row:0.0}   Column {col:0.0}";
        UpdateProbe(at);
    }

    void Stage_MouseLeave(object sender, MouseEventArgs e)
    {
        if (Dragging)
            return;
        HideProbe();
    }

    void Stage_SizeChanged(object sender, SizeChangedEventArgs e)
    {
        if (PendingFit)
            Fit();
    }

    void NotYet_Click(object sender, RoutedEventArgs e)
    {
        string name = sender is MenuItem menu ? menu.Header?.ToString() ?? ""
            : sender is Button button ? button.Content?.ToString() ?? ""
            : "";
        Log(name + " 還沒做。步驟見 docs/csharp-ide-steps.md。");
    }

    void ShowTab_Click(object sender, RoutedEventArgs e)
    {
        if (sender is Button button && button.Tag is string tag)
            ShowTab(tag);
    }

    void ShowTab(string tag)
    {
        PageOps.Visibility = tag == "ops" ? Visibility.Visible : Visibility.Collapsed;
        PageSyntax.Visibility = tag == "syntax" ? Visibility.Visible : Visibility.Collapsed;
        PageLog.Visibility = tag == "log" ? Visibility.Visible : Visibility.Collapsed;
        PageProblems.Visibility = tag == "problems" ? Visibility.Visible : Visibility.Collapsed;
    }

    void OperatorSearch_TextChanged(object sender, TextChangedEventArgs e)
    {
        string query = OperatorSearch.Text.Trim();
        OperatorList.ItemsSource = string.IsNullOrEmpty(query)
            ? Catalog
            : Catalog.Where(item => item.Name.Contains(query, StringComparison.OrdinalIgnoreCase)).ToArray();
    }

    void OperatorList_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        ParameterPanel.Children.Clear();
        if (OperatorList.SelectedItem is not OperatorItem item)
            return;
        ParameterPanel.Children.Add(new TextBlock
        {
            Text = item.Group + "　" + item.Purpose,
            TextWrapping = TextWrapping.Wrap,
            Foreground = (Brush)FindResource("Muted"),
            Margin = new Thickness(0, 0, 0, 8),
        });
        foreach (string name in item.Parameters)
        {
            ParameterPanel.Children.Add(new TextBlock { Text = name, Margin = new Thickness(0, 6, 0, 2) });
            ParameterPanel.Children.Add(new TextBox());
        }
        Log("選了運算子 " + item.Name + "。參數先顯示，還沒執行。");
    }

    void ClearLog_Click(object sender, RoutedEventArgs e) => LogBox.Clear();

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

    void ZoomAtCenter(double factor)
    {
        if (PictureData == null)
            return;
        ZoomAt(Stage.ActualWidth / 2, Stage.ActualHeight / 2, factor);
    }

    void ZoomAt(double sx, double sy, double factor)
    {
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
        ImageScale.ScaleX = scale;
        ImageScale.ScaleY = scale;
        ImageOffset.X = x;
        ImageOffset.Y = y;
        RenderOptions.SetBitmapScalingMode(Picture, scale < 3 ? BitmapScalingMode.Linear : BitmapScalingMode.NearestNeighbor);
        string percent = (scale * 100).ToString("0") + "%";
        ZoomText.Text = percent;
        StatusZoom.Text = percent;
    }

    (double col, double row) ImagePoint(Point screen) => ((screen.X - CamX) / Zoom, (screen.Y - CamY) / Zoom);

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
        string where = $"Row {row:0.0}   Column {col:0.0}";
        string color = "R —  G —  B —";
        if (ic >= 0 && ir >= 0 && ic < PictureData.Width && ir < PictureData.Height)
        {
            int index = (ir * PictureData.Width + ic) * PictureData.Channels;
            if (PictureData.Channels == 1)
            {
                byte gray = PictureData.Pixels[index];
                color = $"R {gray}  G {gray}  B {gray}";
            }
            else
            {
                byte b = PictureData.Pixels[index];
                byte g = PictureData.Pixels[index + 1];
                byte r = PictureData.Pixels[index + 2];
                color = $"R {r}  G {g}  B {b}";
            }
        }
        StatusPix.Text = where + "   " + color;
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
        StatusPix.Text = "Row —   Column —   R —  G —  B —";
    }

    void ShowZoomRect(Point a, Point b)
    {
        Canvas.SetLeft(ZoomRect, Math.Min(a.X, b.X));
        Canvas.SetTop(ZoomRect, Math.Min(a.Y, b.Y));
        ZoomRect.Width = Math.Abs(b.X - a.X);
        ZoomRect.Height = Math.Abs(b.Y - a.Y);
        ZoomRect.Visibility = Visibility.Visible;
    }

    void ApplyMode(GfxMode mode)
    {
        Mode = mode;
        Brush on = (Brush)FindResource("Accent");
        ModeSelect.Background = mode == GfxMode.Select ? on : ModeOff;
        ModePan.Background = mode == GfxMode.Pan ? on : ModeOff;
        ModeZoom.Background = mode == GfxMode.Zoom ? on : ModeOff;
        Stage.Cursor = mode switch
        {
            GfxMode.Pan => Cursors.Hand,
            GfxMode.Zoom => Cursors.Cross,
            _ => Cursors.Arrow,
        };
    }

    enum GfxMode { Select, Pan, Zoom }

    void Log(string message)
    {
        LogBox.AppendText(DateTime.Now.ToString("HH:mm:ss") + "  " + message + Environment.NewLine);
        LogBox.ScrollToEnd();
    }

    static BitmapSource ToBitmap(ImageBuffer image)
    {
        PixelFormat format = image.Channels == 1 ? PixelFormats.Gray8 : PixelFormats.Bgr24;
        int stride = image.Width * image.Channels;
        var source = BitmapSource.Create(image.Width, image.Height, 96, 96, format, null, image.Pixels, stride);
        source.Freeze();
        return source;
    }

    sealed record OperatorItem(string Name, string Group, string Purpose, string[] Parameters);
}
