using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows;
using AvalonDock.Layout;
using AvalonDock.Serializer.Xml;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
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

    string? CurrentPath;
    string SavedText = "";

    string? CurrentProject;
    string? CurrentProjectFolder;
    string? SampleOrigin;
    bool ApplyingTolerance;
    string TolSignature = "";

    readonly ProgramRun Engine = new();
    bool Running;
    GraphicsWindow? GfxFloat;
    OperatorDialog? ParamDialog;
    readonly Dictionary<string, object> DockPanes = new(StringComparer.Ordinal);
    string DefaultLayout = "";
    bool LoadingLayout;
    System.Windows.Threading.DispatcherTimer? LayoutSaveTimer;
    string? DrawTool;
    bool DrawDragging;
    bool DrawReady;
    bool HasDraft;
    double DraftARow;
    double DraftACol;
    double DraftBRow;
    double DraftBCol;

    readonly List<string> RunNotes = [];
    readonly List<Problem> Problems = [];
    bool ShowErrors = true;
    bool ShowWarnings = true;
    bool ShowMessages = true;
    bool SuppressProblemJump;

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
        RefreshProblems();
        SavedText = ProgramFile.Normalize(ProgramText.Text);
        UpdateDirty();
        ApplyFilter();
        BuildOperatorMenu();
        if (Catalog.Length > 0)
        {
            OperatorList.SelectedItem = Catalog[0];
            ShowHelp(Catalog[0]);
        }
        ShowTab("ops");
        Loaded += (_, _) => LoadDockLayout();
        Closing += (_, _) => SaveDockLayout();
        ApplyMode(GfxMode.Select);
        BuildTemplateMenu();
        SampleMenu.SubmenuOpened += (_, _) => FillProgramMenu(SampleMenu.Items, includeWorkspace: false);
        RefreshTolerances();
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
        if (DrawTool != null)
        {
            Point at = e.GetPosition(Stage);
            Stage.CaptureMouse();
            DrawDragging = true;
            DrawReady = false;
            HasDraft = true;
            (DraftACol, DraftARow) = ImagePoint(at);
            DraftBCol = DraftACol;
            DraftBRow = DraftARow;
            ShapeConfirm.Visibility = Visibility.Collapsed;
            UpdateDraft();
            e.Handled = true;
            return;
        }
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

    void Stage_MouseRightDown(object sender, MouseButtonEventArgs e)
    {
        List<string> tools = CurrentDrawTools();
        if (tools.Count == 0)
            return;
        var menu = new ContextMenu { PlacementTarget = Stage };
        foreach (string tool in tools)
        {
            var item = new MenuItem { Header = ShapeGeometry.Title(tool), Tag = tool };
            item.Click += (_, _) => BeginDraw(tool);
            menu.Items.Add(item);
        }
        menu.IsOpen = true;
        e.Handled = true;
    }

    void Stage_MouseMove(object sender, MouseEventArgs e)
    {
        Point at = e.GetPosition(Stage);
        if (DrawDragging && HasDraft)
        {
            (DraftBCol, DraftBRow) = ImagePoint(at);
            UpdateDraft();
            UpdateProbe(at);
            return;
        }
        if (Dragging && PictureData != null && Mode == GfxMode.Pan)
            ApplyView(Zoom, DragCamX + at.X - DragStart.X, DragCamY + at.Y - DragStart.Y);
        else if (Dragging && Mode == GfxMode.Zoom)
            ShowZoomRect(DragStart, at);
        UpdateProbe(at);
    }

    void Stage_MouseUp(object sender, MouseButtonEventArgs e)
    {
        if (DrawDragging)
        {
            DrawDragging = false;
            Stage.ReleaseMouseCapture();
            Point end = e.GetPosition(Stage);
            (DraftBCol, DraftBRow) = ImagePoint(end);
            double dc = DraftBCol - DraftACol;
            double dr = DraftBRow - DraftARow;
            DrawReady = Math.Sqrt(dc * dc + dr * dr) >= 2 && ParamDialog != null;
            UpdateDraft();
            if (DrawReady)
                ShowShapeConfirm();
            else
                ShapeConfirm.Visibility = Visibility.Collapsed;
            return;
        }
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

    void BuildTemplateMenu()
    {
        TemplateMenu.Items.Clear();
        foreach (TemplateEntry entry in ProgramLibrary.Templates)
        {
            var item = new MenuItem { Header = entry.Title };
            item.Click += (_, _) => OpenTemplate(entry);
            TemplateMenu.Items.Add(item);
        }
    }

    void TemplateList_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new TemplateDialog { Owner = this };
        dialog.Confirm = () => ConfirmDiscard(dialog);
        dialog.Opened = entry => OpenTemplate(entry, confirm: false);
        dialog.ShowDialog();
    }

    void Samples_Click(object sender, RoutedEventArgs e)
    {
        var menu = new ContextMenu
        {
            PlacementTarget = BtnSamples,
            Placement = PlacementMode.Bottom,
            Background = (Brush)FindResource("PanelBackground"),
            Foreground = (Brush)FindResource("Ink"),
        };
        FillProgramMenu(menu.Items, includeWorkspace: true);
        menu.IsOpen = true;
    }

    void FillProgramMenu(ItemCollection items, bool includeWorkspace)
    {
        items.Clear();
        string? repo = ProjectStore.FindRepoRoot();
        if (repo == null)
        {
            items.Add(new MenuItem { Header = "找不到 samples", IsEnabled = false });
            return;
        }
        if (includeWorkspace)
            items.Add(new MenuItem { Header = "範例", IsEnabled = false });
        AddProgramItems(items, ProgramLibrary.Samples(repo));
        if (!includeWorkspace)
            return;
        items.Add(new Separator());
        items.Add(new MenuItem { Header = "已儲存", IsEnabled = false });
        AddProgramItems(items, ProgramLibrary.Workspace(repo));
    }

    void AddProgramItems(ItemCollection items, IReadOnlyList<ProgramEntry> programs)
    {
        if (programs.Count == 0)
        {
            items.Add(new MenuItem { Header = "沒有程式", IsEnabled = false });
            return;
        }
        foreach (ProgramEntry entry in programs)
        {
            var item = new MenuItem
            {
                Header = new TextBlock { Text = entry.Title },
                ToolTip = entry.Name,
            };
            item.Click += (_, _) => OpenLibrary(entry, null, confirm: true);
            items.Add(item);
        }
    }

    void OpenTemplate(TemplateEntry entry, bool confirm = true)
    {
        string? repo = ProjectStore.FindRepoRoot();
        if (repo == null)
        {
            StatusLine.Text = "找不到 samples";
            Log(StatusLine.Text);
            return;
        }
        string path = Path.Combine(repo, "samples", entry.File);
        if (!File.Exists(path))
        {
            StatusLine.Text = "找不到程式";
            Log(StatusLine.Text);
            return;
        }
        OpenLibrary(new ProgramEntry(entry.File, entry.Title, path, "samples"), entry.Expect, confirm);
    }

    void OpenLibrary(ProgramEntry entry, string? expect, bool confirm)
    {
        if (confirm && !ConfirmDiscard())
            return;
        try
        {
            string disk = File.ReadAllText(entry.Path);
            string source = entry.Kind == "samples" ? ToleranceEdit.Restore(entry.Name, disk) : disk;
            CurrentProject = null;
            CurrentProjectFolder = null;
            Engine.ProjectFolder = null;
            string? path = entry.Kind == "workspace" ? entry.Path : null;
            LoadSource(source, entry.Name, path, true, null, expect ?? "已開啟 " + entry.Name);
            if (entry.Kind == "samples")
                SampleOrigin = entry.Path;
        }
        catch (Exception ex)
        {
            StatusLine.Text = "開啟失敗：" + ex.Message;
            Log(StatusLine.Text);
        }
    }

    void ViewPane_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not MenuItem item || item.Tag is not string tag)
            return;
        if (item.IsChecked == true)
            ShowPane(tag);
        else
            HidePane(tag);
    }

    void PinAll_Click(object sender, RoutedEventArgs e)
    {
        foreach (LayoutAnchorable pane in Dock.Layout.Descendents().OfType<LayoutAnchorable>().ToList())
        {
            if (pane.IsHidden)
                pane.Show();
            if (pane.IsAutoHidden)
                pane.ToggleAutoHide();
        }
        SyncViewChecks();
        StatusLine.Text = "已全部釘選";
        Log(StatusLine.Text);
    }

    void ResetLayout_Click(object sender, RoutedEventArgs e)
    {
        if (DefaultLayout.Length > 0)
            ApplyLayout(DefaultLayout);
        try
        {
            if (File.Exists(LayoutFile()))
                File.Delete(LayoutFile());
        }
        catch (Exception ex)
        {
            Log("版面檔沒清掉：" + ex.Message);
        }
        GfxFloat?.Close();
        SyncViewChecks();
        StatusLine.Text = "已回復版面";
        Log(StatusLine.Text);
    }

    void LoadDockLayout()
    {
        RememberPanes();
        DefaultLayout = CaptureLayout();
        string path = LayoutFile();
        if (!File.Exists(path))
        {
            SyncViewChecks();
            return;
        }
        try
        {
            ApplyLayout(File.ReadAllText(path));
            if (!HasAllPanes())
                ApplyLayout(DefaultLayout);
        }
        catch (Exception ex)
        {
            Log("版面讀不回來，已用預設：" + ex.Message);
            try { ApplyLayout(DefaultLayout); } catch (Exception) { }
        }
        SyncViewChecks();
    }

    void SaveDockLayout()
    {
        if (LoadingLayout || DefaultLayout.Length == 0)
            return;
        try
        {
            string path = LayoutFile();
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, CaptureLayout());
        }
        catch (Exception ex)
        {
            Log("版面沒存成：" + ex.Message);
        }
    }

    void Dock_LayoutChanged(object? sender, EventArgs e)
    {
        if (LoadingLayout || DefaultLayout.Length == 0)
            return;
        SyncViewChecks();
        LayoutSaveTimer ??= new System.Windows.Threading.DispatcherTimer { Interval = TimeSpan.FromMilliseconds(400) };
        LayoutSaveTimer.Stop();
        LayoutSaveTimer.Tick -= SaveLayoutTick;
        LayoutSaveTimer.Tick += SaveLayoutTick;
        LayoutSaveTimer.Start();
    }

    void SaveLayoutTick(object? sender, EventArgs e)
    {
        LayoutSaveTimer?.Stop();
        SaveDockLayout();
    }

    string CaptureLayout()
    {
        var serializer = new XmlLayoutSerializer(Dock);
        using var writer = new StringWriter();
        serializer.Serialize(writer);
        return writer.ToString();
    }

    void ApplyLayout(string xml)
    {
        RememberPanes();
        LoadingLayout = true;
        try
        {
            var serializer = new XmlLayoutSerializer(Dock);
            serializer.LayoutSerializationCallback += (_, args) =>
            {
                string id = args.Model.ContentId ?? "";
                if (DockPanes.TryGetValue(id, out object? content))
                    args.Content = content;
                else
                    args.Cancel = true;
            };
            using var reader = new StringReader(xml);
            serializer.Deserialize(reader);
        }
        finally
        {
            LoadingLayout = false;
        }
    }

    void RememberPanes()
    {
        if (Dock.Layout == null)
            return;
        foreach (LayoutAnchorable pane in Dock.Layout.Descendents().OfType<LayoutAnchorable>())
        {
            if (!string.IsNullOrEmpty(pane.ContentId) && pane.Content != null)
                DockPanes[pane.ContentId] = pane.Content;
        }
    }

    bool HasAllPanes()
    {
        foreach (string id in new[] { "program", "gfx", "vars", "ops", "syntax", "log", "problems" })
        {
            if (FindPane(id) == null)
                return false;
        }
        return true;
    }

    LayoutAnchorable? FindPane(string id) =>
        Dock.Layout?.Descendents().OfType<LayoutAnchorable>().FirstOrDefault(pane => pane.ContentId == id);

    static string LayoutFile() => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Photonix",
        "layout.xml");

    static string PaneId(string tag) => tag == "bottom" ? "ops" : tag;

    void SyncViewChecks()
    {
        ViewProgram.IsChecked = IsPaneShown("program");
        ViewGfx.IsChecked = IsPaneShown("gfx");
        ViewVars.IsChecked = IsPaneShown("vars");
        ViewOps.IsChecked = IsPaneShown("ops");
    }

    bool IsPaneShown(string id)
    {
        LayoutAnchorable? pane = FindPane(id);
        return pane != null && !pane.IsHidden && !pane.IsAutoHidden;
    }

    void FloatGfx_Click(object sender, RoutedEventArgs e)
    {
        if (GfxFloat != null)
        {
            GfxFloat.Activate();
            NotifyFloat();
            StatusLine.Text = "圖形視窗已拉出";
            return;
        }
        var window = new GraphicsWindow { Owner = this };
        window.Closed += (_, _) =>
        {
            if (ReferenceEquals(GfxFloat, window))
                GfxFloat = null;
        };
        GfxFloat = window;
        window.Show();
        NotifyFloat();
        StatusLine.Text = "圖形視窗已拉出";
        Log(StatusLine.Text);
    }

    void ShowPane(string tag)
    {
        LayoutAnchorable? pane = FindPane(PaneId(tag));
        if (pane == null)
            return;
        if (pane.IsHidden)
            pane.Show();
        if (pane.IsAutoHidden)
            pane.ToggleAutoHide();
        pane.IsSelected = true;
        SyncViewChecks();
    }

    void HidePane(string tag)
    {
        LayoutAnchorable? pane = FindPane(PaneId(tag));
        if (pane == null)
            return;
        if (!pane.IsHidden && !pane.IsAutoHidden)
            pane.ToggleAutoHide();
        SyncViewChecks();
        StatusLine.Text = "已隱藏" + PaneTitle(tag);
    }

    static string PaneTitle(string tag) => tag switch
    {
        "program" => "程式",
        "gfx" => "圖形視窗",
        "vars" => "變數",
        _ => "算子",
    };

    void NotifyFloat() => GfxFloat?.UpdateView(PictureData, Frame, Painted());

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

    void ShowTab(string tag) => ShowPane(tag == "ops" ? "bottom" : tag);

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
        RefreshProblems();
        UpdateDirty();
        RefreshTolerances();
    }

    void RefreshTolerances()
    {
        if (TolItems == null)
            return;
        if (ApplyingTolerance && TolBar.IsKeyboardFocusWithin)
            return;
        string source = ProgramText.Text ?? "";
        string signature = ToleranceSignature(source);
        List<ToleranceLimit> limits = ToleranceEdit.ReadLimits(source);
        List<ToleranceExpect> expects = ToleranceEdit.ReadExpects(source);
        bool teach = ToleranceEdit.ReadPointLists(source).Any(item => item.Name == "WantRow")
            && ToleranceEdit.ReadPointLists(source).Any(item => item.Name == "WantCol");
        if (limits.Count == 0 && expects.Count == 0 && !teach)
        {
            TolBar.Visibility = Visibility.Collapsed;
            TolItems.Children.Clear();
            TolSignature = "";
            return;
        }
        TolBar.Visibility = Visibility.Visible;
        if (signature == TolSignature)
            return;
        TolSignature = signature;
        TolItems.Children.Clear();
        string note = expects.Count > 0 && limits.Count == 0
            ? "預期字會記在這台電腦，改完按執行"
            : "公差會記在這台電腦，改完按執行";
        TolItems.Children.Add(Note(note));
        foreach (ToleranceLimit field in limits)
        {
            var row = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 0, 12, 4) };
            row.Children.Add(Note(field.Label));
            row.Children.Add(TolBox(field.Min, "number", field.MinLine, field.MinName, field.Label + "下限"));
            row.Children.Add(Note("到"));
            row.Children.Add(TolBox(field.Max, "number", field.MaxLine, field.MaxName, field.Label + "上限"));
            TolItems.Children.Add(row);
        }
        foreach (ToleranceExpect field in expects)
        {
            var row = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 0, 12, 4) };
            row.Children.Add(Note(field.Label));
            TextBox box = TolBox(field.Value, "text", field.Line, field.Name, field.Label);
            box.Width = 140;
            row.Children.Add(box);
            TolItems.Children.Add(row);
        }
        if (teach)
            TolItems.Children.Add(TolButton("設為預期", "把這次量到的圓心寫進 WantRow、WantCol", (_, _) => TeachCenters()));
        TolItems.Children.Add(TolButton("預設", "回到程式檔裡的數字", (_, _) => ResetTolerances()));
    }

    string ToleranceSignature(string source)
    {
        var limits = ToleranceEdit.ReadLimits(source).Select(field => field.Stem + "=" + field.Min + ".." + field.Max + "@" + field.MinLine);
        var expects = ToleranceEdit.ReadExpects(source).Select(field => field.Value + "@" + field.Line);
        bool teach = ToleranceEdit.ReadPointLists(source).Any(item => item.Name == "WantRow")
            && ToleranceEdit.ReadPointLists(source).Any(item => item.Name == "WantCol");
        return string.Join("|", limits) + "#" + string.Join("|", expects) + "#" + teach;
    }

    TextBlock Note(string text) => new()
    {
        Text = text,
        Foreground = (Brush)FindResource("Muted"),
        VerticalAlignment = VerticalAlignment.Center,
        Margin = new Thickness(0, 0, 4, 0),
    };

    TextBox TolBox(string value, string kind, int line, string name, string tip)
    {
        var box = new TextBox
        {
            Text = value,
            Width = 64,
            Margin = new Thickness(0, 0, 4, 0),
            ToolTip = tip,
            Tag = new TolEdit(kind, line, name),
        };
        box.LostFocus += (_, _) => CommitTol(box);
        box.KeyDown += (_, keyArgs) =>
        {
            if (keyArgs.Key != Key.Enter)
                return;
            CommitTol(box);
            keyArgs.Handled = true;
        };
        return box;
    }

    Button TolButton(string text, string tip, RoutedEventHandler click)
    {
        var button = new Button { Content = text, Margin = new Thickness(0, 0, 8, 4), ToolTip = tip };
        button.Click += click;
        return button;
    }

    void CommitTol(TextBox box)
    {
        if (ApplyingTolerance || box.Tag is not TolEdit tag)
            return;
        string source = ProgramFile.Normalize(ProgramText.Text);
        string next = tag.Kind == "text"
            ? ToleranceEdit.ApplyExpect(source, tag.Line, box.Text)
            : ToleranceEdit.ApplyLimit(source, tag.Line, tag.Name, box.Text.Trim());
        if (next == source)
        {
            if (tag.Kind == "number" && !LooksNumber(box.Text.Trim()))
            {
                TolSignature = "";
                RefreshTolerances();
            }
            return;
        }
        ApplyingTolerance = true;
        ProgramText.Text = next;
        ApplyingTolerance = false;
        ToleranceEdit.Remember(ProgramName.Text, ProgramText.Text);
        TolSignature = ToleranceSignature(ProgramText.Text ?? "");
    }

    static bool LooksNumber(string text) => Regex.IsMatch(text, @"^-?\d+(?:\.\d+)?$");

    void TeachCenters()
    {
        if (Running)
        {
            StatusLine.Text = "執行中，先等這一輪結束";
            return;
        }
        Engine.Vars.TryGetValue("Row", out object? row);
        Engine.Vars.TryGetValue("Column", out object? column);
        string? next = ToleranceEdit.ApplyPointLists(ProgramText.Text, ToleranceEdit.PointText(row), ToleranceEdit.PointText(column));
        if (next == null)
        {
            StatusLine.Text = "先執行，才有圓心可以寫回";
            return;
        }
        if (ProgramFile.Normalize(next) != ProgramFile.Normalize(ProgramText.Text))
        {
            ApplyingTolerance = true;
            ProgramText.Text = next;
            ApplyingTolerance = false;
            ToleranceEdit.Remember(ProgramName.Text, ProgramText.Text);
            TolSignature = ToleranceSignature(ProgramText.Text ?? "");
        }
        StatusLine.Text = "已把圓心寫進預期位置";
        Log(StatusLine.Text);
    }

    void ResetTolerances()
    {
        bool expects = ToleranceEdit.ReadExpects(ProgramText.Text).Count > 0;
        bool limits = ToleranceEdit.ReadLimits(ProgramText.Text).Count > 0;
        ToleranceEdit.Clear(ProgramName.Text);
        string? file = SampleOrigin ?? CurrentPath;
        if (!string.IsNullOrEmpty(file) && File.Exists(file))
        {
            string disk = ProgramFile.Normalize(File.ReadAllText(file));
            ApplyingTolerance = true;
            ProgramText.Text = disk;
            ApplyingTolerance = false;
            SavedText = ProgramFile.Normalize(ProgramText.Text);
            UpdateDirty();
        }
        TolSignature = "";
        RefreshTolerances();
        StatusLine.Text = expects && !limits ? "已恢復預設預期字" : "已恢復預設上下限";
        Log(StatusLine.Text);
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
        Engine.ProjectFolder = CurrentProjectFolder;
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
        Engine.ProjectFolder = CurrentProjectFolder;
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
        RunNotes.Clear();
        Running = false;
        BtnRun.IsEnabled = true;
        Shapes = [];
        LineTimes.Clear();
        NextLine = 1;
        StatusLine.Text = "已重設";
        Log("已重設");
        RefreshRails();
        RefreshProblems();
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
        RefreshVars();
        if (halt == Halt.Done)
            RunNotes.Add("執行完成（" + Engine.Steps + " 步）");
        else if (halt == Halt.Breakpoint)
            RunNotes.Add("中斷點 第 " + Engine.Pc + " 行");
        else if (halt == Halt.Stopped)
            RunNotes.Add("已停止");
        RefreshProblems();
        if (halt == Halt.Error)
            ShowRunError(Engine.Error ?? "發生錯誤", Engine.ErrorLine);
    }

    void RefreshProblems()
    {
        if (Catalog == null)
            return;
        var found = ProgramProblems.Collect(
            ProgramText.Text ?? "",
            string.IsNullOrWhiteSpace(ProgramName.Text) ? "程式" : ProgramName.Text.Trim(),
            Catalog,
            Engine.Error,
            Engine.ErrorLine,
            RunNotes);
        Problems.Clear();
        Problems.AddRange(found);
        int errors = Problems.Count(item => item.Sev == "error");
        int warnings = Problems.Count(item => item.Sev == "warning");
        int messages = Problems.Count(item => item.Sev == "message");
        FilterErrors.Content = "錯誤 " + errors;
        FilterWarnings.Content = "警告 " + warnings;
        FilterMessages.Content = "訊息 " + messages;
        StatusErrors.Content = "錯誤 " + errors;
        StatusWarnings.Content = "警告 " + warnings;
        StatusMessages.Content = "訊息 " + messages;
        if (FindPane("problems") is LayoutAnchorable problems)
            problems.Title = Problems.Count == 0 ? "錯誤清單" : "錯誤清單 " + Problems.Count;
        PaintFilter(FilterErrors, ShowErrors);
        PaintFilter(FilterWarnings, ShowWarnings);
        PaintFilter(FilterMessages, ShowMessages);
        SuppressProblemJump = true;
        ProblemGrid.ItemsSource = Problems.Where(item =>
            item.Sev == "error" && ShowErrors
            || item.Sev == "warning" && ShowWarnings
            || item.Sev == "message" && ShowMessages).ToList();
        SuppressProblemJump = false;
    }

    void Filter_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not Button button || button.Tag is not string tag)
            return;
        if (tag == "error")
            ShowErrors = !ShowErrors;
        else if (tag == "warning")
            ShowWarnings = !ShowWarnings;
        else if (tag == "message")
            ShowMessages = !ShowMessages;
        RefreshProblems();
    }

    void StatusProblems_Click(object sender, RoutedEventArgs e) => ShowTab("problems");

    void ProblemGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (SuppressProblemJump || ProblemGrid.SelectedItem is not Problem problem || problem.Line <= 0)
            return;
        GoToLine(problem.Line);
    }

    void GoToLine(int line)
    {
        string text = ProgramText.Text ?? "";
        int index = 0;
        for (int i = 1; i < line && index < text.Length; i++)
        {
            int newline = text.IndexOf('\n', index);
            if (newline < 0)
                break;
            index = newline + 1;
        }
        ProgramText.Focus();
        ProgramText.Select(Math.Min(index, text.Length), 0);
        RevealLine(line);
    }

    void ShowRunError(string text, int line)
    {
        string where = line > 0 ? "第 " + line + " 行。" : "";
        MessageBox.Show(this, text + "\n\n" + where + "已經算出的圖和變數還留著", "錯誤", MessageBoxButton.OK, MessageBoxImage.Error);
    }

    static void PaintFilter(Button button, bool on)
    {
        button.Background = on
            ? new SolidColorBrush(Color.FromRgb(0x3D, 0x8B, 0xFD))
            : new SolidColorBrush(Color.FromRgb(0x21, 0x26, 0x2D));
    }

    bool SuppressVars;

    void RefreshVars()
    {
        string? iconicName = (IconicList.SelectedItem as VarRow)?.Name;
        string? controlName = (ControlList.SelectedItem as VarRow)?.Name;
        var iconic = new List<VarRow>();
        var control = new List<VarRow>();
        foreach (VarText item in ProgramVars.Describe(Engine.Vars))
        {
            var row = new VarRow
            {
                Name = item.Name,
                Kind = item.Kind,
                Info = item.Info,
                Full = item.Full,
                Iconic = item.Iconic,
                Thumb = item.Iconic ? ThumbOf(Engine.Vars[item.Name]) : null,
            };
            if (item.Iconic)
                iconic.Add(row);
            else
                control.Add(row);
        }
        SuppressVars = true;
        IconicList.ItemsSource = iconic;
        ControlList.ItemsSource = control;
        IconicEmpty.Visibility = iconic.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
        ControlEmpty.Visibility = control.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
        IconicTitle.Text = iconic.Count == 0 ? "圖像" : "圖像  " + iconic.Count;
        IconicList.SelectedItem = iconic.Find(row => row.Name == iconicName);
        ControlList.SelectedItem = control.Find(row => row.Name == controlName);
        SuppressVars = false;
        if (ControlList.SelectedItem is VarRow picked)
            ShowControl(picked);
    }

    void IconicList_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (SuppressVars || IconicList.SelectedItem is not VarRow row)
            return;
        ControlList.SelectedItem = null;
        PreviewIconic(row);
    }

    void ControlList_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (SuppressVars || ControlList.SelectedItem is not VarRow row)
            return;
        IconicList.SelectedItem = null;
        ShowControl(row);
    }

    void ShowControl(VarRow row) => VarDetail.Text = row.Name + "  " + row.Kind + "\n" + row.Full;

    void PreviewIconic(VarRow row)
    {
        if (!Engine.Vars.TryGetValue(row.Name, out object? value))
            return;
        switch (value)
        {
            case ImageBuffer image:
                ShowRunImage(image);
                Shapes = [];
                break;
            case Region region:
                EnsureBackdrop(region.Width, region.Height);
                Shapes = Operators.Overlays(region);
                break;
            case Region[] regions:
                if (regions.Length > 0)
                    EnsureBackdrop(regions[0].Width, regions[0].Height);
                Shapes = Operators.Overlays(regions);
                break;
            case EdgePoint[][] edges:
                Shapes = EdgeOverlays(edges);
                break;
            default:
                return;
        }
        EmptyHint.Visibility = Visibility.Collapsed;
        Fit();
        StatusLine.Text = "預覽 " + row.Name + " · 再執行會回到程式畫面";
        VarDetail.Text = row.Name + "  " + row.Kind + "\n" + row.Info;
    }

    void EnsureBackdrop(int width, int height)
    {
        if (PictureData != null && PictureData.Width == width && PictureData.Height == height)
            return;
        var blank = new byte[width * height];
        ShowRunImage(ImageBuffer.Gray(width, height, blank));
    }

    void ShowRunImage(ImageBuffer image)
    {
        PictureData = image;
        Frame = ImageFrame.Create(image.Width, image.Height, image.Pixels, image.Channels == 1 ? PixelLayout.Gray : PixelLayout.Bgr);
    }

    static Vizion.Overlay[] EdgeOverlays(EdgePoint[][] edges)
    {
        var overlays = new List<Vizion.Overlay>();
        foreach (EdgePoint[] chain in edges)
        {
            if (chain.Length < 2)
                continue;
            overlays.Add(Vizion.Overlay.Polyline(chain.Select(point => point.Row).ToArray(), chain.Select(point => point.Column).ToArray(), 0xFF00E05A));
        }
        return overlays.ToArray();
    }

    static BitmapSource? ThumbOf(object value)
    {
        BitmapSource? source = value switch
        {
            ImageBuffer image => ImageThumb(image),
            Region region => RegionThumb(region),
            Region[] regions => regions.Length == 0 ? null : RegionThumb(Operators.Union(regions)),
            _ => null,
        };
        source?.Freeze();
        return source;
    }

    static BitmapSource ImageThumb(ImageBuffer image)
    {
        const int edge = 48;
        double scale = Math.Min(1, edge / (double)Math.Max(image.Width, image.Height));
        int width = Math.Max(1, (int)Math.Round(image.Width * scale));
        int height = Math.Max(1, (int)Math.Round(image.Height * scale));
        int channels = image.Channels;
        var pixels = new byte[width * height * channels];
        for (int y = 0; y < height; y++)
        {
            int srcRow = Math.Min(image.Height - 1, (int)(y / scale));
            for (int x = 0; x < width; x++)
            {
                int srcCol = Math.Min(image.Width - 1, (int)(x / scale));
                int from = (srcRow * image.Width + srcCol) * channels;
                int to = (y * width + x) * channels;
                for (int c = 0; c < channels; c++)
                    pixels[to + c] = image.Pixels[from + c];
            }
        }
        PixelFormat format = channels == 1 ? PixelFormats.Gray8 : PixelFormats.Bgr24;
        return BitmapSource.Create(width, height, 96, 96, format, null, pixels, width * channels);
    }

    static BitmapSource RegionThumb(Region region)
    {
        const int edge = 48;
        double scale = Math.Min((double)edge / region.Width, (double)edge / region.Height);
        int width = Math.Max(1, (int)Math.Round(region.Width * scale));
        int height = Math.Max(1, (int)Math.Round(region.Height * scale));
        var pixels = new byte[width * height];
        for (int i = 0; i < region.Rows.Length; i++)
        {
            int y = Math.Clamp((int)(region.Rows[i] * scale), 0, height - 1);
            int x0 = Math.Clamp((int)(region.Columns[i] * scale), 0, width);
            int x1 = Math.Clamp((int)((region.Columns[i] + region.Lengths[i]) * scale), 0, width);
            for (int x = x0; x < x1; x++)
                pixels[y * width + x] = 255;
        }
        return BitmapSource.Create(width, height, 96, 96, PixelFormats.Gray8, null, pixels, width);
    }

    void Project_Click(object sender, RoutedEventArgs e)
    {
        string? repo = ProjectStore.FindRepoRoot();
        if (repo == null)
        {
            StatusLine.Text = "找不到 workspace";
            Log(StatusLine.Text);
            return;
        }
        var dialog = new ProjectDialog(repo, CurrentProject) { Owner = this };
        dialog.ConfirmDiscard = () => ConfirmDiscard();
        dialog.Source = () => ProgramText.Text ?? "";
        dialog.ProgramName = () => ProgramName.Text;
        dialog.Noted = text =>
        {
            StatusLine.Text = text;
            Log(text);
        };
        dialog.Opened = (project, program) =>
        {
            string folder = ProjectStore.Folder(repo, project);
            string source = ProjectStore.Read(repo, project, program);
            CurrentProject = project;
            CurrentProjectFolder = folder;
            Engine.ProjectFolder = folder;
            LoadSource(source, program, Path.Combine(folder, program), true, null);
            StatusLine.Text = "專案 " + project + " · " + program;
        };
        dialog.Saved = (project, path, source, images) =>
        {
            CurrentProject = project;
            CurrentProjectFolder = ProjectStore.Folder(repo, project);
            Engine.ProjectFolder = CurrentProjectFolder;
            if (ProgramFile.Normalize(source) != ProgramFile.Normalize(ProgramText.Text))
                ProgramText.Text = source;
            CurrentPath = path;
            ProgramName.Text = Path.GetFileName(path);
            SavedText = ProgramFile.Normalize(ProgramText.Text);
            UpdateDirty();
            if (images.Count > 0)
                Log("已拷進專案：" + string.Join("、", images));
        };
        dialog.MakeProgram = (project, filename) =>
        {
            string folder = ProjectStore.Folder(repo, project);
            CurrentProject = project;
            CurrentProjectFolder = folder;
            Engine.ProjectFolder = folder;
            LoadSource("* 新程式\n", filename, null, false, null);
            ProjectProgram saved = ProjectStore.Save(repo, project, filename, ProgramText.Text ?? "");
            CurrentPath = saved.Path;
            ProgramName.Text = saved.Name;
            SavedText = ProgramFile.Normalize(ProgramText.Text);
            UpdateDirty();
            StatusLine.Text = "專案 " + project + " · " + saved.Name;
        };
        dialog.Renamed = (previous, renamed) =>
        {
            if (CurrentProject != previous)
                return;
            string? oldFolder = CurrentProjectFolder;
            CurrentProject = renamed;
            CurrentProjectFolder = ProjectStore.Folder(repo, renamed);
            Engine.ProjectFolder = CurrentProjectFolder;
            string prefix = oldFolder == null
                ? ""
                : oldFolder.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
            if (prefix.Length > 1 && CurrentPath != null && CurrentPath.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                CurrentPath = Path.Combine(CurrentProjectFolder, Path.GetFileName(CurrentPath));
        };
        dialog.ShowDialog();
    }

    void NewProgram_Click(object sender, RoutedEventArgs e)
    {
        if (!ConfirmDiscard())
            return;
        LoadSource("* 新程式\n", "untitled.viz", null, clean: false, skipped: null);
    }

    void OpenProgram_Click(object sender, RoutedEventArgs e)
    {
        if (!ConfirmDiscard())
            return;
        var dialog = new OpenFileDialog
        {
            Title = "開啟",
            Filter = "程式 (*.viz;*.hdv;*.hdev;*.hdvp;*.hdpl;*.dev;*.dvp;*.txt)|*.viz;*.hdv;*.hdev;*.hdvp;*.hdpl;*.dev;*.dvp;*.txt|所有檔案 (*.*)|*.*",
        };
        if (dialog.ShowDialog(this) != true)
            return;
        try
        {
            byte[] bytes = File.ReadAllBytes(dialog.FileName);
            string name = Path.GetFileName(dialog.FileName);
            if (ProgramFile.IsHalcon(name))
            {
                (string source, List<string> skipped) = HDevelopFile.Convert(bytes, OperatorNames());
                LoadSource(source, name, dialog.FileName, clean: true, skipped);
            }
            else
                LoadSource(DecodeProgram(bytes), name, dialog.FileName, clean: true, skipped: null);
        }
        catch (Exception ex)
        {
            StatusLine.Text = "開啟失敗：" + ex.Message;
            Log(StatusLine.Text);
        }
    }

    void SaveProgram_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrEmpty(CurrentPath))
        {
            SaveProgramAs_Click(sender, e);
            return;
        }
        WriteProgram(CurrentPath);
    }

    void SaveProgramAs_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new SaveFileDialog
        {
            Title = "另存新檔",
            FileName = ProgramFile.EnsureName(ProgramName.Text),
            Filter = "Photonix (*.viz)|*.viz|HDV (*.hdv)|*.hdv|文字 (*.txt)|*.txt|HDevelop (*.hdev)|*.hdev|程序 (*.hdvp)|*.hdvp|程序庫 (*.hdpl)|*.hdpl|所有檔案 (*.*)|*.*",
        };
        if (dialog.ShowDialog(this) != true)
            return;
        string path = dialog.FileName;
        if (!ProgramFile.Suffixes.Any(suffix => path.EndsWith(suffix, StringComparison.OrdinalIgnoreCase)))
            path += ".viz";
        if (WriteProgram(path))
            CurrentPath = path;
    }

    void ExportCs_Click(object sender, RoutedEventArgs e)
    {
        string className = CsharpExport.ClassName(ProgramName.Text);
        string code;
        try
        {
            code = CsharpExport.Export(ProgramText.Text, className, Catalog);
        }
        catch (Exception ex)
        {
            StatusLine.Text = ex.Message;
            Log(ex.Message);
            MessageBox.Show(this, ex.Message, "匯出 C#", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        var dialog = new SaveFileDialog
        {
            Title = "匯出 C#",
            FileName = className + ".cs",
            Filter = "C# (*.cs)|*.cs|所有檔案 (*.*)|*.*",
        };
        if (dialog.ShowDialog(this) != true)
            return;
        try
        {
            File.WriteAllText(dialog.FileName, code, new UTF8Encoding(false));
            StatusLine.Text = "已匯出 " + Path.GetFileName(dialog.FileName);
            Log(StatusLine.Text);
        }
        catch (Exception ex)
        {
            StatusLine.Text = "匯出失敗：" + ex.Message;
            Log(StatusLine.Text);
        }
    }

    bool WriteProgram(string path)
    {
        try
        {
            string text = ProgramFile.ForDisk(ProgramText.Text, path);
            File.WriteAllText(path, text, new UTF8Encoding(false));
            CurrentPath = path;
            ProgramName.Text = Path.GetFileName(path);
            SavedText = ProgramFile.Normalize(ProgramText.Text);
            StatusLine.Text = "已儲存 " + path;
            Log(StatusLine.Text);
            UpdateDirty();
            return true;
        }
        catch (Exception ex)
        {
            StatusLine.Text = "存檔失敗：" + ex.Message;
            Log(StatusLine.Text);
            return false;
        }
    }

    void LoadSource(string source, string name, string? path, bool clean, IReadOnlyList<string>? skipped, string? note = null)
    {
        SampleOrigin = null;
        TolSignature = "";
        source = ProgramFile.Normalize(source);
        SavedText = clean ? source : "";
        CurrentPath = path;
        ProgramName.Text = name;
        ProgramText.Text = source;
        Breakpoints.Clear();
        RunNotes.Clear();
        Engine.Reset();
        NextLine = 1;
        PictureData = null;
        Frame = null;
        Picture.Source = null;
        Picture.Visibility = Visibility.Collapsed;
        EmptyHint.Visibility = Visibility.Visible;
        Shapes = [];
        LineTimes.Clear();
        NotifyFloat();
        RefreshRails();
        RefreshVars();
        RefreshProblems();
        UpdateDirty();
        if (skipped != null && skipped.Count > 0)
        {
            string skippedNote = skipped.Count > 4
                ? "已開啟 " + name + "。" + skipped.Count + " 個有密碼的程序沒有打開"
                : "已開啟。有密碼的程序沒有打開：" + string.Join("、", skipped);
            StatusLine.Text = skippedNote;
            Log(skippedNote);
        }
        else
        {
            StatusLine.Text = note ?? (path == null ? "新程式" : "已開啟 " + name);
            Log(StatusLine.Text);
        }
    }

    bool ConfirmDiscard(Window? owner = null)
    {
        if (!IsDirty())
            return true;
        return MessageBox.Show(owner ?? this, "程式還沒存。要放棄這些修改嗎？", "Photonix", MessageBoxButton.YesNo, MessageBoxImage.Warning) == MessageBoxResult.Yes;
    }

    bool IsDirty() => ProgramFile.Normalize(ProgramText.Text) != ProgramFile.Normalize(SavedText);

    void UpdateDirty()
    {
        bool dirty = IsDirty();
        string name = string.IsNullOrWhiteSpace(ProgramName.Text) ? "untitled.viz" : ProgramName.Text.Trim();
        Title = (dirty ? "* " : "") + name + " — Photonix";
        if (BtnSave != null)
            BtnSave.Content = dirty ? "儲存 *" : "儲存";
    }

    void ProgramName_TextChanged(object sender, TextChangedEventArgs e)
    {
        UpdateDirty();
        RefreshProblems();
    }

    Dictionary<string, string> OperatorNames()
    {
        var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (OperatorDoc op in Catalog)
        {
            string bare = op.Name.Contains('.') ? op.Name[(op.Name.IndexOf('.') + 1)..] : op.Name;
            map[bare] = op.Name;
        }
        return map;
    }

    static string DecodeProgram(byte[] raw)
    {
        if (raw.Length >= 2 && raw[0] == 0xFF && raw[1] == 0xFE)
            return Encoding.Unicode.GetString(raw);
        if (raw.Length >= 2 && raw[0] == 0xFE && raw[1] == 0xFF)
            return Encoding.BigEndianUnicode.GetString(raw);
        if (raw.Length >= 3 && raw[0] == 0xEF && raw[1] == 0xBB && raw[2] == 0xBF)
            return Encoding.UTF8.GetString(raw, 3, raw.Length - 3);
        return Encoding.UTF8.GetString(raw);
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
        if (Keyboard.Modifiers == ModifierKeys.Control && Keyboard.Modifiers != (ModifierKeys.Control | ModifierKeys.Alt))
        {
            if (e.Key == Key.N)
            {
                NewProgram_Click(this, e);
                e.Handled = true;
                return;
            }
            if (e.Key == Key.O)
            {
                OpenProgram_Click(this, e);
                e.Handled = true;
                return;
            }
            if (e.Key == Key.S)
            {
                SaveProgram_Click(this, e);
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
        OperatorDialog? previous = ParamDialog;
        ParamDialog = null;
        previous?.Close();
        var dialog = new OperatorDialog(op, values, editing) { Owner = this };
        dialog.Accepted += result =>
        {
            string call = OperatorCall.FormatCall(op.Name, result);
            Memory.Remember(op, result);
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
        };
        dialog.Closed += (_, _) =>
        {
            if (!ReferenceEquals(ParamDialog, dialog))
                return;
            ParamDialog = null;
            CancelDraw();
            RefreshShapeTools();
        };
        ParamDialog = dialog;
        dialog.Show();
        RefreshShapeTools();
    }

    void DrawTool_Click(object sender, RoutedEventArgs e)
    {
        if (sender is Button button && button.Tag is string tool)
            BeginDraw(tool);
    }

    void BeginDraw(string tool)
    {
        if (ParamDialog == null || !CurrentDrawTools().Contains(tool))
            return;
        DrawTool = tool;
        DrawDragging = false;
        DrawReady = false;
        HasDraft = false;
        ShapeConfirm.Visibility = Visibility.Collapsed;
        UpdateDraft();
        RefreshShapeTools();
        StatusLine.Text = ShapeGeometry.Title(tool) + "：在圖上拖曳，起點到終點的方向會保留";
    }

    void ShapeApply_Click(object sender, RoutedEventArgs e)
    {
        if (ParamDialog == null || DrawTool == null || !DrawReady)
            return;
        ParamDialog.ApplyShape(DrawTool, DraftARow, DraftACol, DraftBRow, DraftBCol);
        StatusLine.Text = "已填入" + ShapeGeometry.Title(DrawTool);
        CancelDraw();
    }

    void ShapeCancel_Click(object sender, RoutedEventArgs e) => CancelDraw();

    void CancelDraw()
    {
        DrawTool = null;
        DrawDragging = false;
        DrawReady = false;
        HasDraft = false;
        if (ShapeConfirm != null)
            ShapeConfirm.Visibility = Visibility.Collapsed;
        UpdateDraft();
        RefreshShapeTools();
    }

    void ShowShapeConfirm()
    {
        if (ParamDialog == null || DrawTool == null)
            return;
        ShapeConfirmText.Text = ParamDialog.ShapeText(DrawTool, DraftARow, DraftACol, DraftBRow, DraftBCol);
        ShapeConfirm.Visibility = Visibility.Visible;
    }

    List<string> CurrentDrawTools() => ParamDialog == null ? [] : ShapeGeometry.Tools(ParamDialog.GeometryNames);

    void RefreshShapeTools()
    {
        if (DrawLine == null)
            return;
        List<string> tools = CurrentDrawTools();
        PaintDraw(DrawLine, "line", tools);
        PaintDraw(DrawRect, "rect", tools);
        PaintDraw(DrawCircle, "circle", tools);
        PaintDraw(DrawEllipse, "ellipse", tools);
        if (DrawTool != null && !tools.Contains(DrawTool))
            CancelDraw();
    }

    void PaintDraw(Button button, string tool, List<string> tools)
    {
        button.Visibility = tools.Contains(tool) ? Visibility.Visible : Visibility.Collapsed;
        button.Background = DrawTool == tool ? (Brush)FindResource("Accent") : ModeOff;
    }

    void UpdateDraft()
    {
        if (DraftLine == null)
            return;
        DraftLine.Visibility = Visibility.Collapsed;
        DraftRect.Visibility = Visibility.Collapsed;
        DraftEllipse.Visibility = Visibility.Collapsed;
        if (!HasDraft || DrawTool == null)
            return;
        Point a = Screen(DraftACol, DraftARow);
        Point b = Screen(DraftBCol, DraftBRow);
        if (DrawTool == "line")
        {
            DraftLine.X1 = a.X;
            DraftLine.Y1 = a.Y;
            DraftLine.X2 = b.X;
            DraftLine.Y2 = b.Y;
            DraftLine.Visibility = Visibility.Visible;
            return;
        }
        if (DrawTool == "circle")
        {
            double radius = Math.Sqrt((b.X - a.X) * (b.X - a.X) + (b.Y - a.Y) * (b.Y - a.Y));
            Canvas.SetLeft(DraftEllipse, a.X - radius);
            Canvas.SetTop(DraftEllipse, a.Y - radius);
            DraftEllipse.Width = radius * 2;
            DraftEllipse.Height = radius * 2;
            DraftEllipse.Visibility = Visibility.Visible;
            return;
        }
        double left = Math.Min(a.X, b.X);
        double top = Math.Min(a.Y, b.Y);
        double width = Math.Abs(b.X - a.X);
        double height = Math.Abs(b.Y - a.Y);
        if (DrawTool == "ellipse")
        {
            Canvas.SetLeft(DraftEllipse, left);
            Canvas.SetTop(DraftEllipse, top);
            DraftEllipse.Width = Math.Max(1, width);
            DraftEllipse.Height = Math.Max(1, height);
            DraftEllipse.Visibility = Visibility.Visible;
            return;
        }
        Canvas.SetLeft(DraftRect, left);
        Canvas.SetTop(DraftRect, top);
        DraftRect.Width = Math.Max(1, width);
        DraftRect.Height = Math.Max(1, height);
        DraftRect.Visibility = Visibility.Visible;
    }

    Point Screen(double col, double row) => new(CamX + col * Zoom, CamY + row * Zoom);

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
        NotifyFloat();
        if (PictureData == null || Frame == null)
        {
            UpdateDraft();
            return;
        }
        int viewW = (int)Math.Round(Stage.ActualWidth);
        int viewH = (int)Math.Round(Stage.ActualHeight);
        if (viewW < 2 || viewH < 2)
        {
            UpdateDraft();
            return;
        }
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
        UpdateDraft();
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

    sealed record TolEdit(string Kind, int Line, string Name);

    void Log(string message)
    {
        LogBox.AppendText(DateTime.Now.ToString("HH:mm:ss") + "  " + message + Environment.NewLine);
        LogBox.ScrollToEnd();
    }

    sealed class VarRow
    {
        public string Name { get; init; } = "";
        public string Kind { get; init; } = "";
        public string Info { get; init; } = "";
        public string Full { get; init; } = "";
        public bool Iconic { get; init; }
        public ImageSource? Thumb { get; init; }
    }
}
