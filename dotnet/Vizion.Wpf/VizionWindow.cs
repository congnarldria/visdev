using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Vizion;

namespace Vizion.Wpf;

/// <summary>
/// WPF 影像視窗。直接畫在 WPF 畫面裡，沒有 HALCON 那種 HWND 空氣層。
/// Show 一次就顯示，滾輪放大、拖曳平移、雙擊回到整張。連續畫面只留最後一張。
/// </summary>
public sealed class VizionWindow : FrameworkElement
{
    private readonly VizionViewCore _core = new();
    private WriteableBitmap? _bitmap;
    private bool _dragging;
    private Point _drag;
    private double _dpiX = 1;
    private double _dpiY = 1;

    public VizionWindow()
    {
        SnapsToDevicePixels = true;
        UseLayoutRounding = true;
        Focusable = true;
        _core.FrameReady += () => Dispatcher.BeginInvoke(new Action(InvalidateVisual));
    }

    public long Presented => _core.Presented;
    public long Coalesced => _core.Coalesced;
    public int ImageWidth => _core.ImageWidth;
    public int ImageHeight => _core.ImageHeight;

    public event EventHandler<ImagePointerEventArgs>? ImagePointerMove;

    public void Show(int width, int height, byte[] gray, params Overlay[] overlays) =>
        _core.Show(ImageFrame.Gray(width, height, gray), overlays);

    public void Show(int width, int height, byte[] pixels, PixelLayout layout, params Overlay[] overlays) =>
        _core.Show(ImageFrame.Create(width, height, pixels, layout), overlays);

    public void SetOverlays(params Overlay[] overlays) => _core.SetOverlays(overlays);

    public void Fit() => _core.Refit();

    protected override void OnRenderSizeChanged(SizeChangedInfo sizeInfo)
    {
        base.OnRenderSizeChanged(sizeInfo);
        UpdateDpi();
        _core.Resize(Pixels(sizeInfo.NewSize.Width, _dpiX), Pixels(sizeInfo.NewSize.Height, _dpiY));
    }

    protected override void OnRender(DrawingContext drawingContext)
    {
        drawingContext.DrawRectangle(new SolidColorBrush(Color.FromRgb(30, 30, 30)), null, new Rect(RenderSize));
        if (!_core.TryTakeFront(out byte[] pixels, out int width, out int height) || width <= 0 || height <= 0)
            return;
        if (_bitmap == null || _bitmap.PixelWidth != width || _bitmap.PixelHeight != height)
            _bitmap = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgra32, null);
        _bitmap.WritePixels(new Int32Rect(0, 0, width, height), pixels, width * 4, 0);
        drawingContext.DrawImage(_bitmap, new Rect(RenderSize));
    }

    protected override void OnMouseWheel(MouseWheelEventArgs e)
    {
        base.OnMouseWheel(e);
        UpdateDpi();
        var point = e.GetPosition(this);
        _core.ZoomAt(Pixels(point.X, _dpiX), Pixels(point.Y, _dpiY), e.Delta > 0 ? 1.25 : 0.8);
        e.Handled = true;
    }

    protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
    {
        base.OnMouseLeftButtonDown(e);
        _dragging = true;
        _drag = e.GetPosition(this);
        CaptureMouse();
        Focus();
    }

    protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e)
    {
        base.OnMouseLeftButtonUp(e);
        _dragging = false;
        ReleaseMouseCapture();
    }

    protected override void OnMouseMove(MouseEventArgs e)
    {
        base.OnMouseMove(e);
        UpdateDpi();
        var point = e.GetPosition(this);
        if (_dragging && e.LeftButton == MouseButtonState.Pressed)
        {
            _core.PanPixels(Pixels(point.X - _drag.X, _dpiX), Pixels(point.Y - _drag.Y, _dpiY));
            _drag = point;
        }
        if (_core.TryHit(Pixels(point.X, _dpiX), Pixels(point.Y, _dpiY), out double row, out double column, out int gray))
            ImagePointerMove?.Invoke(this, new ImagePointerEventArgs(row, column, gray));
    }

    protected override void OnMouseDown(MouseButtonEventArgs e)
    {
        base.OnMouseDown(e);
        if (e.ClickCount == 2 && e.ChangedButton == MouseButton.Left)
            Fit();
    }

    private void UpdateDpi()
    {
        var source = PresentationSource.FromVisual(this);
        if (source?.CompositionTarget == null)
            return;
        _dpiX = source.CompositionTarget.TransformToDevice.M11;
        _dpiY = source.CompositionTarget.TransformToDevice.M22;
        if (_dpiX <= 0)
            _dpiX = 1;
        if (_dpiY <= 0)
            _dpiY = 1;
    }

    private static int Pixels(double dips, double dpi) => (int)Math.Round(dips * dpi);
}
