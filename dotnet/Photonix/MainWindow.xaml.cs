using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using Vizion;

namespace Photonix;

public partial class MainWindow : Window
{
    readonly OperatorDoc[] Catalog;
    readonly OperatorMemory Memory;

    ImageBuffer? PictureData;
    ImageFrame? Frame;
    Overlay[] Shapes = [];
    int LineWidth = 2;
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

    int SenseIndex;
    int SenseTokenStart;
    bool SenseForce;
    string SenseLabel = "";
    List<SenseItem> SenseItems = [];

    int NextLine = 1;
    readonly HashSet<int> Breakpoints = [];
    readonly Dictionary<int, (double Ms, int Count)> LineTimes = [];

    readonly ProgramRun Engine = new();
    bool Running;

    public MainWindow()
    {
        InitializeComponent();
        Catalog = OperatorCatalog.Load();
        Memory = OperatorMemory.Load();
        ProgramHighlight.Attach(ProgramText, Catalog.Select(op => op.Name));
        LineGutter.Attach(ProgramText);
        TimeRail.Attach(ProgramText);
        LineGutter.LineClick += (_, line) => ToggleBreakpoint(line);
        LineGutter.ArrowMoved += (_, line) =>
        {
            NextLine = line;
            Engine.MoveTo(line);
            StatusLine.Text = "下一行是第 " + line + " 行";
            RefreshRails();
        };
        RefreshRails();
        ApplyFilter();
        BuildOperatorMenu();
        if (Catalog.Length > 0)
        {
            OperatorList.SelectedItem = Catalog[0];
            ShowHelp(Catalog[0]);
        }
        ShowTab("ops");
        ApplyMode(GfxMode.Select);
        Log("Photonix 已開啟。版面對齊網頁版，底層是 Vizion.Operators。");
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
            Frame = ImageFrame.Create(
                PictureData.Width,
                PictureData.Height,
                PictureData.Pixels,
                PictureData.Channels == 1 ? PixelLayout.Gray : PixelLayout.Bgr);
            Shapes = [];
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
        else
            Present();
    }

    void ContourWidth_TextChanged(object sender, TextChangedEventArgs e)
    {
        if (int.TryParse(ContourWidth.Text.Trim(), out int value) && value >= 1 && value <= 10 && value != LineWidth)
        {
            LineWidth = value;
            Present();
        }
    }

    void ContourWidth_LostFocus(object sender, RoutedEventArgs e)
    {
        if (!int.TryParse(ContourWidth.Text.Trim(), out int value))
            value = LineWidth;
        value = Math.Clamp(value, 1, 10);
        LineWidth = value;
        ContourWidth.Text = value.ToString();
        Present();
    }

    void SaveView_Click(object sender, RoutedEventArgs e)
    {
        if (PictureData == null || Frame == null)
        {
            StatusLine.Text = "還沒有畫面可以存";
            Log("還沒有畫面可以存");
            return;
        }
        var dialog = new SaveFileDialog
        {
            Filter = "PNG (*.png)|*.png",
            FileName = "view.png",
            Title = "存畫面",
        };
        if (dialog.ShowDialog(this) != true)
            return;
        int width = PictureData.Width;
        int height = PictureData.Height;
        var pixels = new byte[width * height * 4];
        var map = new ScreenMap(0, 0, width, height, 0, 0, height, width);
        FrameComposer.Render(Frame, map, Painted(), pixels, width, height);
        var source = BitmapSource.Create(width, height, 96, 96, PixelFormats.Bgra32, null, pixels, width * 4);
        source.Freeze();
        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(source));
        using (var stream = File.Create(dialog.FileName))
            encoder.Save(stream);
        StatusLine.Text = "已存畫面 " + dialog.FileName;
        Log("已存畫面 " + dialog.FileName + "，含底圖和疊加。");
    }

    void About_Click(object sender, RoutedEventArgs e)
    {
        MessageBox.Show(
            this,
            "Photonix\n視覺開發環境。底層呼叫 Vizion.Operators。",
            "關於 Photonix",
            MessageBoxButton.OK,
            MessageBoxImage.Information);
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

    void OperatorSearch_TextChanged(object sender, TextChangedEventArgs e) => ApplyFilter();

    void ApplyFilter()
    {
        string query = OperatorSearch.Text.Trim();
        IEnumerable<OperatorDoc> visible = Catalog;
        if (query.Length > 0)
        {
            visible = Catalog.Where(op => string.Concat(op.Name, op.Purpose, op.Principle, op.Setup, op.Note)
                .Contains(query, StringComparison.OrdinalIgnoreCase));
        }
        var view = new ListCollectionView(visible.ToList());
        view.GroupDescriptions.Add(new PropertyGroupDescription(nameof(OperatorDoc.Group)));
        OperatorList.ItemsSource = view;
    }

    void BuildOperatorMenu()
    {
        OpMenu.Items.Clear();
        var recent = Memory.Recent.Where(name => Catalog.Any(op => op.Name == name)).Take(8).ToList();
        if (recent.Count > 0)
        {
            var recentMenu = new MenuItem { Header = "最近" };
            foreach (string name in recent)
                recentMenu.Items.Add(OperatorMenuItem(Catalog.First(op => op.Name == name)));
            OpMenu.Items.Add(recentMenu);
        }
        MenuItem? groupItem = null;
        string? group = null;
        foreach (OperatorDoc op in Catalog)
        {
            if (op.Group != group)
            {
                group = op.Group;
                groupItem = new MenuItem { Header = group };
                OpMenu.Items.Add(groupItem);
            }
            groupItem!.Items.Add(OperatorMenuItem(op));
        }
    }

    MenuItem OperatorMenuItem(OperatorDoc op)
    {
        var item = new MenuItem { Header = op.Name, ToolTip = op.Short, Tag = op };
        item.Click += OperatorMenu_Click;
        return item;
    }

    void OperatorMenu_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not MenuItem { Tag: OperatorDoc op })
            return;
        if (!string.IsNullOrEmpty(OperatorSearch.Text))
            OperatorSearch.Text = "";
        OperatorList.SelectedItem = op;
        OperatorList.ScrollIntoView(op);
        ShowTab("ops");
        ShowHelp(op);
        OpenParameterDialog(op, null, null);
    }

    void ProgramText_MouseUp(object sender, MouseButtonEventArgs e)
    {
        if (ProgramText.SelectionLength != 0)
            return;
        string text = ProgramText.Text ?? "";
        int pos = Math.Clamp(ProgramText.SelectionStart, 0, text.Length);
        int lineStart = pos == 0 ? 0 : text.LastIndexOf('\n', pos - 1) + 1;
        int lineEnd = text.IndexOf('\n', pos);
        if (lineEnd < 0)
            lineEnd = text.Length;
        string line = text[lineStart..lineEnd].TrimEnd('\r');
        ParsedCall? parsed = OperatorCall.ParseLine(line);
        if (parsed == null)
            return;
        int nameAt = line.IndexOf(parsed.Value.Name, StringComparison.Ordinal);
        if (nameAt < 0)
            return;
        int col = pos - lineStart;
        if (col < nameAt || col > nameAt + parsed.Value.Name.Length)
            return;
        OperatorDoc? op = FindOperator(parsed.Value.Name);
        if (op == null)
            return;
        int lineIndex = text[..lineStart].Count(ch => ch == '\n');
        CloseSense();
        OpenParameterDialog(op, lineIndex, parsed.Value.Args);
    }

    void Undo_Click(object sender, RoutedEventArgs e)
    {
        ProgramText.Focus();
        ProgramText.Undo();
    }

    void Redo_Click(object sender, RoutedEventArgs e)
    {
        ProgramText.Focus();
        ProgramText.Redo();
    }

    void Cut_Click(object sender, RoutedEventArgs e)
    {
        ProgramText.Focus();
        ProgramText.Cut();
    }

    void Copy_Click(object sender, RoutedEventArgs e)
    {
        ProgramText.Focus();
        ProgramText.Copy();
    }

    void Paste_Click(object sender, RoutedEventArgs e)
    {
        ProgramText.Focus();
        ProgramText.Paste();
    }

    void SelectAll_Click(object sender, RoutedEventArgs e)
    {
        ProgramText.Focus();
        ProgramText.SelectAll();
    }

    void ProgramText_TextChanged(object sender, TextChangedEventArgs e)
    {
        RefreshRails();
        UpdateSense(false);
    }

    void ToggleBreakpoint(int line)
    {
        if (!Breakpoints.Add(line))
            Breakpoints.Remove(line);
        RefreshRails();
    }

    void RefreshRails()
    {
        int count = ProgramCounter.LineCount(ProgramText.Text ?? "");
        Breakpoints.RemoveWhere(line => line < 1 || line > count);
        if (NextLine > count)
            NextLine = count;
        LineGutter.CurrentLine = NextLine;
        LineGutter.Breakpoints = Breakpoints;
        TimeRail.CurrentLine = NextLine;
        TimeRail.Times = LineTimes;
        ProgramHighlight.CurrentLine = NextLine;
        LineGutter.InvalidateVisual();
        TimeRail.InvalidateVisual();
        ProgramHighlight.InvalidateVisual();
    }

    void Run_Click(object sender, RoutedEventArgs e)
    {
        if (Running)
            return;
        CloseSense();
        if (Engine.Done || Engine.Error != null)
            Engine.Reset();
        Engine.Cancel = false;
        Running = true;
        BtnRun.IsEnabled = false;
        Pump();
    }

    void Step_Click(object sender, RoutedEventArgs e)
    {
        if (Running)
            return;
        CloseSense();
        if (Engine.Error != null)
        {
            StatusLine.Text = Engine.Error;
            NextLine = Engine.ErrorLine;
            RefreshRails();
            return;
        }
        Halt halt = Engine.Step(SourceLines(), TraceBox.IsChecked == true);
        Publish(halt);
    }

    void Stop_Click(object sender, RoutedEventArgs e)
    {
        Engine.Cancel = true;
        if (!Running)
        {
            StatusLine.Text = "已停止";
            Log("已停止");
        }
    }

    void Reset_Click(object sender, RoutedEventArgs e)
    {
        Engine.Reset();
        Running = false;
        BtnRun.IsEnabled = true;
        Shapes = [];
        LineTimes.Clear();
        NextLine = 1;
        StatusLine.Text = "已重設";
        Log("已重設");
        RefreshRails();
        Present();
    }

    void Trace_Click(object sender, RoutedEventArgs e) => TraceBox.IsChecked = TraceBox.IsChecked != true;

    void Pump()
    {
        Halt halt = Engine.Run(SourceLines(), Breakpoints, TraceBox.IsChecked == true, 40);
        if (halt == Halt.Yield && !Engine.Cancel)
        {
            Publish(Halt.Yield);
            Dispatcher.BeginInvoke(Pump, System.Windows.Threading.DispatcherPriority.Background);
            return;
        }
        Running = false;
        BtnRun.IsEnabled = true;
        if (Engine.Cancel)
            halt = Halt.Stopped;
        Publish(halt);
    }

    void Publish(Halt halt)
    {
        foreach (string message in Engine.Messages)
            Log(message);
        Engine.Messages.Clear();
        LineTimes.Clear();
        foreach (var pair in Engine.Times)
            LineTimes[pair.Key] = pair.Value;
        if (Engine.Image != null)
        {
            PictureData = Engine.Image;
            Frame = ImageFrame.Create(
                Engine.Image.Width,
                Engine.Image.Height,
                Engine.Image.Pixels,
                Engine.Image.Channels == 1 ? PixelLayout.Gray : PixelLayout.Bgr);
            EmptyHint.Visibility = Visibility.Collapsed;
        }
        Shapes = Engine.Shapes;
        NextLine = Engine.Error != null ? Engine.ErrorLine : Engine.Done ? 0 : Engine.Pc;
        if (halt == Halt.Error)
            StatusLine.Text = Engine.Error ?? "";
        else if (halt == Halt.Breakpoint)
            StatusLine.Text = "停在第 " + Engine.Pc + " 行";
        else if (halt == Halt.Done)
            StatusLine.Text = "執行完成（" + Engine.Steps + " 步）";
        else if (halt == Halt.Stopped)
            StatusLine.Text = "已停止";
        else if (halt == Halt.None || halt == Halt.Yield)
            StatusLine.Text = "第 " + Math.Max(1, Engine.Pc) + " 行";
        if (NextLine > 0 && halt is Halt.Error or Halt.Breakpoint or Halt.None)
            RevealLine(NextLine);
        if (PictureData != null)
            Present();
        RefreshRails();
    }

    string[] SourceLines() => (ProgramText.Text ?? "").Replace("\r\n", "\n").Replace('\r', '\n').Split('\n');

    protected override void OnPreviewKeyDown(KeyEventArgs e)
    {
        if (Keyboard.Modifiers == ModifierKeys.None)
        {
            if (e.Key == Key.F5)
            {
                Run_Click(this, e);
                e.Handled = true;
                return;
            }
            if (e.Key == Key.F6)
            {
                Step_Click(this, e);
                e.Handled = true;
                return;
            }
            if (e.Key == Key.F7)
            {
                Reset_Click(this, e);
                e.Handled = true;
                return;
            }
        }
        base.OnPreviewKeyDown(e);
    }

    void RevealLine(int line)
    {
        string text = ProgramText.Text ?? "";
        int index = 0;
        for (int i = 1; i < line; i++)
        {
            int newline = text.IndexOf('\n', index);
            if (newline < 0)
                return;
            index = newline + 1;
        }
        Rect rect = ProgramText.GetRectFromCharacterIndex(Math.Min(index, text.Length));
        if (ProgramText.Template.FindName("PART_ContentHost", ProgramText) is ScrollViewer scroll)
        {
            double view = scroll.ViewportHeight;
            if (view > 0 && (rect.Y < 0 || rect.Bottom > view))
                scroll.ScrollToVerticalOffset(Math.Max(0, scroll.VerticalOffset + rect.Y - (view - rect.Height) / 2));
        }
    }

    void ProgramText_SelectionChanged(object sender, RoutedEventArgs e) => UpdateSense(false);

    void ProgramText_PreviewKeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Space && Keyboard.Modifiers == ModifierKeys.Control)
        {
            SenseForce = true;
            UpdateSense(true);
            e.Handled = true;
            return;
        }
        if (!SensePopup.IsOpen)
            return;
        if (e.Key == Key.Down)
        {
            MoveSense(1);
            e.Handled = true;
        }
        else if (e.Key == Key.Up)
        {
            MoveSense(-1);
            e.Handled = true;
        }
        else if (e.Key == Key.Tab)
        {
            AcceptSense();
            e.Handled = true;
        }
        else if (e.Key == Key.Escape)
        {
            CloseSense();
            e.Handled = true;
        }
    }

    void UpdateSense(bool force)
    {
        if (!IsLoaded)
            return;
        string text = ProgramText.Text ?? "";
        int start = ProgramText.SelectionStart;
        int end = start + ProgramText.SelectionLength;
        SenseChunk? chunk = ProgramSyntax.Chunk(text, start, end);
        if (chunk == null)
        {
            CloseSense();
            return;
        }
        var built = ProgramSyntax.Suggest(chunk.Value, Catalog, text, force || SenseForce);
        SenseForce = false;
        if (built.Items.Count == 0)
        {
            CloseSense();
            return;
        }
        int keep = built.Items.ToList().FindIndex(item => item.Label == SenseLabel);
        SenseItems = built.Items.ToList();
        SenseIndex = keep >= 0 ? keep : 0;
        SenseLabel = SenseItems[SenseIndex].Label;
        SenseTokenStart = chunk.Value.TokenStart;
        SenseSig.Text = built.Signature;
        SenseSig.Visibility = built.Signature.Length == 0 ? Visibility.Collapsed : Visibility.Visible;
        SenseList.ItemsSource = SenseItems;
        SenseList.SelectedIndex = SenseIndex;
        Rect caret = ProgramText.GetRectFromCharacterIndex(start);
        SensePopup.HorizontalOffset = caret.X;
        SensePopup.VerticalOffset = caret.Bottom;
        SensePopup.IsOpen = true;
    }

    void MoveSense(int step)
    {
        if (SenseItems.Count == 0)
            return;
        SenseIndex = (SenseIndex + step + SenseItems.Count) % SenseItems.Count;
        SenseLabel = SenseItems[SenseIndex].Label;
        SenseList.SelectedIndex = SenseIndex;
        SenseList.ScrollIntoView(SenseItems[SenseIndex]);
    }

    void AcceptSense()
    {
        if (SenseIndex < 0 || SenseIndex >= SenseItems.Count)
            return;
        SenseItem item = SenseItems[SenseIndex];
        int end = ProgramText.SelectionStart;
        int start = Math.Clamp(SenseTokenStart, 0, end);
        ProgramText.Select(start, end - start);
        ProgramText.SelectedText = item.Insert;
        int caret = start + (item.Caret ?? item.Insert.Length);
        ProgramText.Select(caret, 0);
        SenseLabel = "";
        CloseSense();
        UpdateSense(false);
    }

    void SenseList_MouseDown(object sender, MouseButtonEventArgs e)
    {
        if (e.OriginalSource is not DependencyObject source)
            return;
        if (ItemsControl.ContainerFromElement(SenseList, source) is not ListBoxItem { Content: SenseItem item })
            return;
        SenseIndex = SenseItems.IndexOf(item);
        SenseLabel = item.Label;
        AcceptSense();
        e.Handled = true;
    }

    void CloseSense()
    {
        SenseItems = [];
        SensePopup.IsOpen = false;
        SenseLabel = "";
    }

    void OpenParameterDialog(OperatorDoc op, int? lineIndex, string[]? preset)
    {
        bool editing = lineIndex != null;
        Memory.Fields.TryGetValue(op.Name, out Dictionary<string, string>? saved);
        string[] values = OperatorCall.DialogValues(op, preset, editing, saved);
        var dialog = new OperatorDialog(op, values, editing) { Owner = this };
        if (dialog.ShowDialog() != true || dialog.Values == null)
            return;
        string call = OperatorCall.FormatCall(op.Name, dialog.Values);
        Memory.Remember(op, dialog.Values);
        Memory.Save();
        if (editing)
        {
            ReplaceProgramLine(lineIndex!.Value, call);
            Log("已改第 " + (lineIndex.Value + 1) + " 行");
        }
        else
            InsertAtCursor(Commented(op, call));
        BuildOperatorMenu();
        ProgramText.Focus();
    }

    OperatorDoc? FindOperator(string token)
    {
        string lower = token.ToLowerInvariant();
        return Catalog.FirstOrDefault(op =>
        {
            string name = op.Name.ToLowerInvariant();
            return name == lower || name == "vizion." + lower;
        });
    }

    void OperatorList_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (OperatorList.SelectedItem is OperatorDoc op)
            ShowHelp(op);
    }

    void OperatorList_MouseDoubleClick(object sender, MouseButtonEventArgs e)
    {
        if (e.OriginalSource is not DependencyObject source)
            return;
        if (ItemsControl.ContainerFromElement(OperatorList, source) is ListBoxItem { Content: OperatorDoc op })
            InsertExample(op);
    }

    void ShowHelp(OperatorDoc op)
    {
        OperatorHelp.Children.Clear();
        Brush muted = (Brush)FindResource("Muted");
        OperatorHelp.Children.Add(Heading(op.Name, 16, FontWeights.Bold));
        if (!string.IsNullOrWhiteSpace(op.Note))
            AddSection("註解", op.Note, muted);
        AddSection("用途", op.Purpose, muted);
        AddSection("原理", op.Principle, muted);
        AddSection("如何設定", op.Setup, muted);
        if (op.Params.Count > 0)
        {
            OperatorHelp.Children.Add(Heading("參數", 13, FontWeights.SemiBold));
            foreach (OperatorParam param in op.Params)
            {
                OperatorHelp.Children.Add(new TextBlock
                {
                    Text = param.Name + "　" + param.Role,
                    Margin = new Thickness(0, 6, 0, 0),
                });
                OperatorHelp.Children.Add(new TextBlock
                {
                    Text = param.Desc,
                    TextWrapping = TextWrapping.Wrap,
                    Foreground = muted,
                });
            }
        }
        AddSection("範例用法", op.Prototype + "\n" + op.Example, muted);
        var insert = new Button { Content = "插入範本", HorizontalAlignment = HorizontalAlignment.Left, Margin = new Thickness(0, 8, 0, 0) };
        insert.Click += (_, _) => InsertExample(op);
        OperatorHelp.Children.Add(insert);
    }

    void AddSection(string title, string body, Brush muted)
    {
        OperatorHelp.Children.Add(Heading(title, 13, FontWeights.SemiBold));
        OperatorHelp.Children.Add(new TextBlock
        {
            Text = body,
            TextWrapping = TextWrapping.Wrap,
            Margin = new Thickness(0, 2, 0, 8),
            Foreground = muted,
        });
    }

    static TextBlock Heading(string text, double size, FontWeight weight) => new()
    {
        Text = text,
        FontSize = size,
        FontWeight = weight,
        TextWrapping = TextWrapping.Wrap,
        Margin = new Thickness(0, 4, 0, 2),
    };

    void InsertExample(OperatorDoc op) => InsertAtCursor(Commented(op, op.Example ?? ""));

    static string Commented(OperatorDoc op, string call)
    {
        if (string.IsNullOrWhiteSpace(op.Note))
            return call;
        string comment = string.Join("\n", op.Note.Replace("\r\n", "\n").Split('\n').Select(line => "* " + line));
        return comment + "\n" + call;
    }

    void ReplaceProgramLine(int lineIndex, string call)
    {
        string text = ProgramText.Text ?? "";
        int start = 0;
        for (int i = 0; i < lineIndex; i++)
        {
            int newline = text.IndexOf('\n', start);
            if (newline < 0)
                return;
            start = newline + 1;
        }
        int end = text.IndexOf('\n', start);
        if (end < 0)
            end = text.Length;
        string current = text[start..end].TrimEnd('\r');
        int indent = 0;
        while (indent < current.Length && char.IsWhiteSpace(current[indent]))
            indent++;
        ProgramText.Select(start, end - start);
        ProgramText.SelectedText = current[..indent] + call;
    }

    void InsertAtCursor(string text)
    {
        string value = ProgramText.Text ?? "";
        int pos = Math.Clamp(ProgramText.SelectionStart, 0, value.Length);
        int start = value.Length == 0 || pos == 0 ? 0 : value.LastIndexOf('\n', pos - 1) + 1;
        string insert = text + "\n";
        ProgramText.Select(start, 0);
        ProgramText.SelectedText = insert;
        ProgramText.Select(start + insert.Length, 0);
        ProgramText.Focus();
        Log("已插入 " + text.Split('\n')[^1]);
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
        string percent = (scale * 100).ToString("0") + "%";
        ZoomText.Text = percent;
        StatusZoom.Text = percent;
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
        FrameComposer.Render(Frame, map, Painted(), pixels, viewW, viewH);
        var source = BitmapSource.Create(viewW, viewH, 96, 96, PixelFormats.Bgra32, null, pixels, viewW * 4);
        source.Freeze();
        Picture.Source = source;
        Picture.Visibility = Visibility.Visible;
    }

    Overlay[] Painted()
    {
        if (Shapes.Length == 0)
            return Shapes;
        var painted = new Overlay[Shapes.Length];
        for (int i = 0; i < Shapes.Length; i++)
        {
            Overlay item = Shapes[i];
            painted[i] = item.Kind is OverlayKind.Polyline or OverlayKind.Cross or OverlayKind.Rectangle
                ? item.WithStroke(LineWidth)
                : item;
        }
        return painted;
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

}
