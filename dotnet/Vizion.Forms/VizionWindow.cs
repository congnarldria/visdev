using System.Runtime.InteropServices;
using Vizion;

namespace Vizion.Forms;

/// <summary>
/// WinForms 影像視窗。Show 一次就顯示，滑鼠滾輪放大、拖曳平移、雙擊回到整張。
/// 合成在背景執行緒，只畫看得到的範圍；畫面來得比畫得還快時，中間的幀會丟掉。
/// </summary>
public sealed class VizionWindow : Control
{
    private readonly VizionViewCore _core = new();
    private Bitmap? _bitmap;
    private bool _dragging;
    private Point _drag;

    public VizionWindow()
    {
        SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint | ControlStyles.OptimizedDoubleBuffer | ControlStyles.ResizeRedraw, true);
        BackColor = Color.FromArgb(30, 30, 30);
        _core.FrameReady += OnFrameReady;
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

    protected override void OnHandleCreated(EventArgs e)
    {
        base.OnHandleCreated(e);
        _core.Resize(ClientSize.Width, ClientSize.Height);
    }

    protected override void OnResize(EventArgs e)
    {
        base.OnResize(e);
        _core.Resize(ClientSize.Width, ClientSize.Height);
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        e.Graphics.Clear(BackColor);
        if (!_core.TryTakeFront(out byte[] pixels, out int width, out int height))
            return;
        if (width != ClientSize.Width || height != ClientSize.Height || width <= 0 || height <= 0)
            return;
        EnsureBitmap(width, height);
        var data = _bitmap!.LockBits(new Rectangle(0, 0, width, height), System.Drawing.Imaging.ImageLockMode.WriteOnly, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
        try
        {
            int rowBytes = width * 4;
            for (int row = 0; row < height; row++)
                Marshal.Copy(pixels, row * rowBytes, IntPtr.Add(data.Scan0, row * data.Stride), rowBytes);
        }
        finally
        {
            _bitmap.UnlockBits(data);
        }
        e.Graphics.DrawImageUnscaled(_bitmap, 0, 0);
    }

    protected override void OnMouseWheel(MouseEventArgs e)
    {
        base.OnMouseWheel(e);
        _core.ZoomAt(e.X, e.Y, e.Delta > 0 ? 1.25 : 0.8);
    }

    protected override void OnMouseDown(MouseEventArgs e)
    {
        base.OnMouseDown(e);
        if (e.Button != MouseButtons.Left)
            return;
        _dragging = true;
        _drag = e.Location;
        Capture = true;
    }

    protected override void OnMouseUp(MouseEventArgs e)
    {
        base.OnMouseUp(e);
        _dragging = false;
        Capture = false;
    }

    protected override void OnMouseMove(MouseEventArgs e)
    {
        base.OnMouseMove(e);
        if (_dragging)
        {
            _core.PanPixels(e.X - _drag.X, e.Y - _drag.Y);
            _drag = e.Location;
        }
        if (_core.TryHit(e.X, e.Y, out double row, out double column, out int gray))
            ImagePointerMove?.Invoke(this, new ImagePointerEventArgs(row, column, gray));
    }

    protected override void OnMouseDoubleClick(MouseEventArgs e)
    {
        base.OnMouseDoubleClick(e);
        if (e.Button == MouseButtons.Left)
            Fit();
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _bitmap?.Dispose();
        base.Dispose(disposing);
    }

    private void OnFrameReady()
    {
        if (!IsHandleCreated || IsDisposed)
            return;
        try
        {
            BeginInvoke(new Action(Invalidate));
        }
        catch (InvalidOperationException)
        {
        }
    }

    private void EnsureBitmap(int width, int height)
    {
        if (_bitmap != null && _bitmap.Width == width && _bitmap.Height == height)
            return;
        _bitmap?.Dispose();
        _bitmap = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
    }
}
