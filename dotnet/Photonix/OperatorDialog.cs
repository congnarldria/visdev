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
    readonly TextBlock Preview = new() { TextWrapping = TextWrapping.Wrap, FontFamily = new System.Windows.Media.FontFamily("Consolas"), Margin = new Thickness(0, 8, 0, 0) };
    readonly OperatorDoc Op;

    public string[]? Values { get; private set; }

    public OperatorDialog(OperatorDoc op, string[] values, bool editing)
    {
        Op = op;
        Title = op.Name;
        Width = 560;
        MaxHeight = 720;
        SizeToContent = SizeToContent.Height;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        Background = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(0x14, 0x17, 0x1C));
        Foreground = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(0xE6, 0xED, 0xF3));
        var muted = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(0x8B, 0x94, 0x9E));

        var form = new StackPanel { Margin = new Thickness(16) };
        form.Children.Add(new TextBlock { Text = op.Purpose, TextWrapping = TextWrapping.Wrap, Foreground = muted, Margin = new Thickness(0, 0, 0, 8) });
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
        form.Children.Add(Preview);

        var ok = new Button { Content = editing ? "更新" : "插入", MinWidth = 72, Margin = new Thickness(8, 0, 0, 0), IsDefault = true };
        ok.Click += Ok_Click;
        var cancel = new Button { Content = "取消", MinWidth = 72, IsCancel = true };
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

        var input = new TextBox { Text = current, MinWidth = 240 };
        input.TextChanged += (_, _) => RefreshPreview();
        Reads.Add(() => input.Text.Trim());
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
        Values = values;
        DialogResult = true;
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
