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

    public MainWindow()
    {
        InitializeComponent();
        OperatorList.ItemsSource = Catalog;
        Log("VisDev 已開啟。底層是 Vizion.Operators。");
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
        try
        {
            PictureData = Operators.ReadImage(dialog.FileName);
            Picture.Source = ToBitmap(PictureData);
            Picture.Visibility = Visibility.Visible;
            EmptyHint.Visibility = Visibility.Collapsed;
            Fit();
            Log($"載入 {dialog.FileName}，{PictureData.Width}×{PictureData.Height}，{PictureData.Channels} 通道。");
        }
        catch (Exception ex)
        {
            Log(ex.Message);
        }
    }

    void Fit_Click(object sender, RoutedEventArgs e) => Fit();

    void OneToOne_Click(object sender, RoutedEventArgs e) => SetZoom(1);

    void ZoomIn_Click(object sender, RoutedEventArgs e) => SetZoom(Zoom * 1.25);

    void ZoomOut_Click(object sender, RoutedEventArgs e) => SetZoom(Zoom / 1.25);

    void Viewport_MouseWheel(object sender, MouseWheelEventArgs e)
    {
        if (PictureData == null)
            return;
        SetZoom(Zoom * (e.Delta > 0 ? 1.12 : 1 / 1.12));
        e.Handled = true;
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

    void Fit()
    {
        if (PictureData == null || Viewport.ActualWidth <= 1 || Viewport.ActualHeight <= 1)
        {
            SetZoom(1);
            return;
        }
        double fit = Math.Min(Viewport.ActualWidth / PictureData.Width, Viewport.ActualHeight / PictureData.Height);
        SetZoom(fit);
    }

    void SetZoom(double zoom)
    {
        Zoom = Math.Clamp(zoom, 0.05, 32);
        ImageScale.ScaleX = Zoom;
        ImageScale.ScaleY = Zoom;
        ZoomText.Text = (Zoom * 100).ToString("0") + "%";
    }

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
