namespace Vizion;

/// <summary>
/// 兩種控制項共用的顯示狀態。工作執行緒合成畫面，畫面太快進來時只留最後一張。
/// </summary>
public sealed class VizionViewCore
{
    private readonly object _gate = new();
    private readonly ViewState _view = new();
    private ImageFrame? _frame;
    private Overlay[] _overlays = Array.Empty<Overlay>();
    private PaintJob? _pending;
    private byte[]? _front;
    private int _frontWidth;
    private int _frontHeight;
    private int _viewWidth;
    private int _viewHeight;
    private int _busy;
    private long _presented;
    private long _coalesced;

    public ViewState View => _view;
    public long Presented => Interlocked.Read(ref _presented);
    public long Coalesced => Interlocked.Read(ref _coalesced);
    public int ImageWidth => _frame?.Width ?? 0;
    public int ImageHeight => _frame?.Height ?? 0;

    public event Action? FrameReady;

    public void Show(ImageFrame frame, IReadOnlyList<Overlay>? overlays = null)
    {
        lock (_gate)
        {
            if (_pending != null)
                Interlocked.Increment(ref _coalesced);
            _frame = frame;
            if (overlays != null)
                _overlays = overlays as Overlay[] ?? overlays.ToArray();
            _pending = new PaintJob(frame, _overlays, _viewWidth, _viewHeight, CopyView());
        }
        Kick();
    }

    public void SetOverlays(IReadOnlyList<Overlay> overlays)
    {
        lock (_gate)
        {
            _overlays = overlays as Overlay[] ?? overlays.ToArray();
            if (_frame == null)
                return;
            _pending = new PaintJob(_frame, _overlays, _viewWidth, _viewHeight, CopyView());
        }
        Kick();
    }

    public void Resize(int width, int height)
    {
        lock (_gate)
        {
            _viewWidth = Math.Max(0, width);
            _viewHeight = Math.Max(0, height);
            if (_frame == null)
                return;
            _pending = new PaintJob(_frame, _overlays, _viewWidth, _viewHeight, CopyView());
        }
        Kick();
    }

    public void Refit()
    {
        lock (_gate)
        {
            _view.Fit();
            if (_frame == null)
                return;
            _pending = new PaintJob(_frame, _overlays, _viewWidth, _viewHeight, CopyView());
        }
        Kick();
    }

    public void ZoomAt(int x, int y, double factor)
    {
        lock (_gate)
        {
            if (_frame == null)
                return;
            var map = _view.Map(_frame.Width, _frame.Height, _viewWidth, _viewHeight);
            if (!map.TryImage(x, y, out double row, out double column))
            {
                row = (_view.Fitted ? _frame.Height / 2.0 : (_view.Row1 + _view.Row2) / 2);
                column = (_view.Fitted ? _frame.Width / 2.0 : (_view.Column1 + _view.Column2) / 2);
            }
            _view.ZoomAt(row, column, factor, _frame.Width, _frame.Height, _viewWidth, _viewHeight);
            _pending = new PaintJob(_frame, _overlays, _viewWidth, _viewHeight, CopyView());
        }
        Kick();
    }

    public void PanPixels(int dx, int dy)
    {
        lock (_gate)
        {
            if (_frame == null)
                return;
            var map = _view.Map(_frame.Width, _frame.Height, _viewWidth, _viewHeight);
            _view.PanPixels(dx, dy, map, _frame.Width, _frame.Height);
            _pending = new PaintJob(_frame, _overlays, _viewWidth, _viewHeight, CopyView());
        }
        Kick();
    }

    public bool TryHit(int x, int y, out double row, out double column, out int gray)
    {
        row = column = 0;
        gray = -1;
        lock (_gate)
        {
            if (_frame == null)
                return false;
            var map = _view.Map(_frame.Width, _frame.Height, _viewWidth, _viewHeight);
            if (!map.TryImage(x, y, out row, out column))
                return false;
            gray = _frame.GrayAt((int)Math.Floor(row), (int)Math.Floor(column));
            return true;
        }
    }

    public bool TryTakeFront(out byte[] pixels, out int width, out int height)
    {
        lock (_gate)
        {
            if (_front == null)
            {
                pixels = Array.Empty<byte>();
                width = height = 0;
                return false;
            }
            pixels = _front;
            width = _frontWidth;
            height = _frontHeight;
            return true;
        }
    }

    private ViewSnapshot CopyView() => new(_view.Fitted, _view.Row1, _view.Column1, _view.Row2, _view.Column2);

    private void Kick()
    {
        if (Interlocked.CompareExchange(ref _busy, 1, 0) != 0)
            return;
        ThreadPool.UnsafeQueueUserWorkItem(delegate { PaintQueued(); }, null);
    }

    private void PaintQueued()
    {
        var again = false;
        try
        {
            while (true)
            {
                PaintJob? job = null;
                lock (_gate)
                {
                    job = _pending;
                    _pending = null;
                }
                if (job == null || job.ViewWidth <= 0 || job.ViewHeight <= 0)
                    break;
                var map = job.View.Fitted
                    ? ScreenMap.Fit(job.Frame.Width, job.Frame.Height, job.ViewWidth, job.ViewHeight)
                    : ScreenMap.Fill(job.ViewWidth, job.ViewHeight, job.View.Row1, job.View.Column1, job.View.Row2, job.View.Column2);
                var pixels = new byte[job.ViewWidth * job.ViewHeight * 4];
                FrameComposer.Render(job.Frame, map, job.Overlays, pixels, job.ViewWidth, job.ViewHeight);
                lock (_gate)
                {
                    _front = pixels;
                    _frontWidth = job.ViewWidth;
                    _frontHeight = job.ViewHeight;
                }
                Interlocked.Increment(ref _presented);
            }
        }
        finally
        {
            Interlocked.Exchange(ref _busy, 0);
            lock (_gate)
                again = _pending != null;
        }
        FrameReady?.Invoke();
        if (again)
            Kick();
    }

    private sealed class PaintJob(ImageFrame frame, Overlay[] overlays, int viewWidth, int viewHeight, ViewSnapshot view)
    {
        public ImageFrame Frame { get; } = frame;
        public Overlay[] Overlays { get; } = overlays;
        public int ViewWidth { get; } = viewWidth;
        public int ViewHeight { get; } = viewHeight;
        public ViewSnapshot View { get; } = view;
    }

    private readonly record struct ViewSnapshot(bool Fitted, double Row1, double Column1, double Row2, double Column2);
}
