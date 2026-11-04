using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using Microsoft.Win32;

namespace Photonix;

/// <summary>填算子參數。確認後由主視窗改那一行，或插進游標處。</summary>
sealed class OperatorDialog : Window
{
    readonly List<Func<string>> Reads = [];
    readonly List<OperatorParam> Fields = [];
    readonly Dictionary<string, TextBox> GeomBoxes = new(StringComparer.Ordinal);
    readonly TextBlock Preview = new() { TextWrapping = TextWrapping.Wrap, FontFamily = new System.Windows.Media.FontFamily("Consolas"), Margin = new Thickness(0, 8, 0, 0) };
    readonly OperatorDoc Op;

    public event Action<string[]>? Accepted;

    public IReadOnlyCollection<string> GeometryNames => GeomBoxes.Keys;

    public OperatorDialog(OperatorDoc op, string[] values, bool editing)
    {
        Op = op;
        Title = op.Name;
        Width = 460;
        MaxHeight = 720;
        SizeToContent = SizeToContent.Height;
        WindowStartupLocation = WindowStartupLocation.Manual;
        ShowInTaskbar = false;
        Background = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(0x14, 0x17, 0x1C));
        Foreground = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(0xE6, 0xED, 0xF3));
        var muted = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(0x8B, 0x94, 0x9E));

        var form = new StackPanel { Margin = new Thickness(16) };
        AddBlock(form, "用途", op.Purpose, muted);
        AddBlock(form, "原理", op.Principle, muted);
        AddBlock(form, "用法", op.Example, muted);
        int afterHelp = form.Children.Count;
        string role = "";
        for (int i = 0; i < op.Params.Count; i++)
        {
            OperatorParam param = op.Params[i];
            if (param.Role != role)
            {
                role = param.Role;
                form.Children.Add(new TextBlock { Text = role, FontWeight = FontWeights.SemiBold, Margin = new Thickness(0, 10, 0, 4) });
            }
            string current = i < values.Length ? values[i] : "";
            form.Children.Add(new TextBlock { Text = param.Name, Margin = new Thickness(0, 6, 0, 2) });
            form.Children.Add(Field(param, current));
            form.Children.Add(new TextBlock { Text = param.Desc, TextWrapping = TextWrapping.Wrap, Foreground = muted });
            Fields.Add(param);
        }
        if (op.Params.Count == 0)
            form.Children.Add(new TextBlock { Text = "這個算子沒有參數。", Foreground = muted });
        if (ShapeGeometry.Tools(GeomBoxes.Keys).Count > 0)
        {
            form.Children.Insert(afterHelp, new TextBlock
            {
                Text = "圖形列的線、框選、圓、橢圓，或在圖上按右鍵。拖曳後按確認填入。起點到終點的方向會保留。",
                TextWrapping = TextWrapping.Wrap,
                Foreground = muted,
                Margin = new Thickness(0, 8, 0, 0),
            });
        }
        form.Children.Add(Preview);

        var ok = new Button { Content = editing ? "更新" : "插入", MinWidth = 72, Margin = new Thickness(8, 0, 0, 0), IsDefault = true };
        ok.Click += Ok_Click;
        var cancel = new Button { Content = "取消", MinWidth = 72 };
        cancel.Click += (_, _) => Close();
        PreviewKeyDown += (_, keyArgs) =>
        {
            if (keyArgs.Key == System.Windows.Input.Key.Escape)
                Close();
        };
        Loaded += (_, _) => PlaceBesideOwner();
        var buttons = new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right, Margin = new Thickness(16, 0, 16, 16) };
        buttons.Children.Add(cancel);
        buttons.Children.Add(ok);

        var root = new DockPanel();
        DockPanel.SetDock(buttons, Dock.Bottom);
        root.Children.Add(buttons);
        root.Children.Add(new ScrollViewer { Content = form, VerticalScrollBarVisibility = ScrollBarVisibility.Auto });
        Content = root;
        RefreshPreview();
    }

    static void AddBlock(StackPanel form, string title, string body, System.Windows.Media.Brush muted)
    {
        if (string.IsNullOrWhiteSpace(body))
            return;
        form.Children.Add(new TextBlock { Text = title, FontWeight = FontWeights.SemiBold, Margin = new Thickness(0, 8, 0, 2) });
        form.Children.Add(new TextBlock { Text = body, TextWrapping = TextWrapping.Wrap, Foreground = muted });
    }

    FrameworkElement Field(OperatorParam param, string current)
    {
        if (param.Role == "控制輸入" && param.Choices.Count > 0)
        {
            bool known = param.Choices.Contains(current);
            var options = new List<string>();
            if (!known && current.Length > 0)
                options.Add(current);
            options.AddRange(param.Choices);
            string picked = known ? current : options[0];
            var box = new ComboBox { MinWidth = 240 };
            foreach (string choice in options)
                box.Items.Add(new ComboBoxItem { Content = Label(choice), Tag = choice });
            box.SelectedIndex = options.IndexOf(picked);
            box.SelectionChanged += (_, _) => RefreshPreview();
            Reads.Add(() => box.SelectedItem is ComboBoxItem item ? item.Tag as string ?? "" : "");
            return box;
        }

        var input = new TextBox { Text = current, MinWidth = 220 };
        input.TextChanged += (_, _) => RefreshPreview();
        Reads.Add(() => input.Text.Trim());
        if (param.Role == "控制輸入" && ShapeGeometry.Names.Contains(param.Name))
            GeomBoxes[param.Name] = input;
        if (string.IsNullOrEmpty(param.Browse))
            return input;
        var browse = new Button { Content = "選檔", Margin = new Thickness(8, 0, 0, 0) };
        browse.Click += (_, _) => Browse(param.Browse, input);
        var row = new StackPanel { Orientation = Orientation.Horizontal };
        row.Children.Add(input);
        row.Children.Add(browse);
        return row;
    }

    void Browse(string kind, TextBox input)
    {
        var dialog = new OpenFileDialog
        {
            Title = "選檔",
            Filter = kind == "plate"
                ? "標定板 (*.cpd;*.descr)|*.cpd;*.descr|所有檔案 (*.*)|*.*"
                : "影像 (*.png;*.jpg;*.bmp;*.tif)|*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff|所有檔案 (*.*)|*.*",
        };
        if (dialog.ShowDialog(this) != true)
            return;
        string path = dialog.FileName.Replace('\\', '/');
        if (path.Length == 0 || path.Contains('\'') || path.Contains('\n') || path.Contains('\r'))
        {
            MessageBox.Show(this, "這個路徑沒辦法放進程式", Op.Name, MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        input.Text = "'" + path + "'";
    }

    void Ok_Click(object sender, RoutedEventArgs e)
    {
        var values = new string[Reads.Count];
        for (int i = 0; i < Reads.Count; i++)
        {
            string value = Reads[i]();
            OperatorParam param = Fields[i];
            if (value.Length == 0)
            {
                MessageBox.Show(this, param.Name + " 還沒有填", Op.Name, MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
            if (param.Role != "控制輸入" && !Regex.IsMatch(value, @"^[A-Za-z_][A-Za-z0-9_]*$"))
            {
                MessageBox.Show(this, param.Name + " 必須是變數名稱", Op.Name, MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
            values[i] = value;
        }
        Accepted?.Invoke(values);
        Close();
    }

    public void ApplyShape(string tool, double aRow, double aCol, double bRow, double bCol)
    {
        bool inner = GeomBoxes.ContainsKey("Radius1") && (Desc("Radius1") ?? "").Contains('內');
        string mode = tool == "rect" && GeomBoxes.ContainsKey("Row1") ? "corners" : inner ? "inner" : "axis";
        Dictionary<string, double> numbers = ShapeGeometry.Numbers(tool, aRow, aCol, bRow, bCol, mode);
        var keep = new HashSet<string>(numbers.Keys, StringComparer.Ordinal);
        foreach ((string name, double value) in numbers)
        {
            if (GeomBoxes.TryGetValue(name, out TextBox? box))
                box.Text = ShapeGeometry.Format(value);
        }
        foreach ((string name, TextBox box) in GeomBoxes)
        {
            if (keep.Contains(name))
                continue;
            OperatorParam? param = Fields.FirstOrDefault(item => item.Role == "控制輸入" && item.Name == name);
            if (param?.Default is "''" or "")
                box.Text = "''";
        }
    }

    public string ShapeText(string tool, double aRow, double aCol, double bRow, double bCol) =>
        ShapeGeometry.Describe(tool, aRow, aCol, bRow, bCol, GeomBoxes.ContainsKey("Row1"));

    string? Desc(string name) => Fields.FirstOrDefault(item => item.Name == name)?.Desc;

    void PlaceBesideOwner()
    {
        if (Owner == null)
            return;
        double width = ActualWidth > 1 ? ActualWidth : Width;
        double left = Owner.Left + Math.Max(0, Owner.ActualWidth - width - 24);
        double top = Owner.Top + 72;
        Left = Math.Max(Owner.Left + 8, left);
        Top = Math.Max(8, top);
    }

    void RefreshPreview()
    {
        if (Reads.Count != Fields.Count)
            return;
        Preview.Text = OperatorCall.FormatCall(Op.Name, Reads.ConvertAll(read => read()));
    }

    static string Label(string choice)
    {
        if (choice.Length >= 2 && choice[0] == '\'' && choice[^1] == '\'')
            return choice[1..^1];
        return choice;
    }
}
