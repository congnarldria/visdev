using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace Photonix;

/// <summary>產線統計和明細。點不合格那一列會打開那張圖。</summary>
static class LineBoard
{
    public static void ShowStats(Window owner, LineReport.Stats stats)
    {
        var body = new StackPanel { Margin = new Thickness(0, 8, 0, 0) };
        if (stats.NoJudge)
        {
            body.Children.Add(new TextBlock { Text = "這份明細沒有判定欄", TextWrapping = TextWrapping.Wrap });
        }
        else
        {
            var grid = new Grid { Margin = new Thickness(0, 0, 0, 8) };
            grid.ColumnDefinitions.Add(new ColumnDefinition());
            grid.ColumnDefinitions.Add(new ColumnDefinition());
            AddStat(grid, 0, "總數", stats.Total.ToString());
            AddStat(grid, 1, "合格", stats.Ok.ToString());
            AddStat(grid, 2, "不合格", stats.Ng.ToString());
            AddStat(grid, 3, "良率", stats.Yield.Length == 0 ? "" : stats.Yield + "%");
            body.Children.Add(grid);
            if (stats.Total > 0)
            {
                var bar = new Grid { Height = 10, Margin = new Thickness(0, 0, 0, 8) };
                bar.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(Math.Max(stats.Ok, 0), GridUnitType.Star) });
                bar.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(Math.Max(stats.Ng, 0), GridUnitType.Star) });
                var ok = new Border { Background = new SolidColorBrush(Color.FromRgb(0x3D, 0xDC, 0x84)) };
                var ng = new Border { Background = new SolidColorBrush(Color.FromRgb(0xFF, 0x5D, 0x4A)) };
                Grid.SetColumn(ng, 1);
                bar.Children.Add(ok);
                bar.Children.Add(ng);
                body.Children.Add(bar);
            }
            if (stats.LastNg.Length > 0)
                body.Children.Add(new TextBlock { Text = "最近不合格 " + stats.LastNg, Margin = new Thickness(0, 4, 0, 0) });
        }
        Show(owner, "產線統計", stats.File, body, null);
    }

    public static void ShowCsv(Window owner, string repo, string csvFile, Action<string> openNg)
    {
        bool onlyNg = false;
        var note = new TextBlock { Foreground = new SolidColorBrush(Color.FromRgb(0x8B, 0x98, 0xAB)), TextWrapping = TextWrapping.Wrap };
        var list = new StackPanel();
        var toggle = new Button { Margin = new Thickness(0, 0, 8, 0) };
        void Reload()
        {
            LineReport.Sheet sheet = LineReport.ReadSheet(repo, csvFile, onlyNg);
            note.Text = LineReport.Note(sheet);
            toggle.Content = sheet.OnlyNg ? "看全部" : "只看不合格";
            list.Children.Clear();
            list.Children.Add(Row(sheet.Header, null));
            for (int i = 0; i < sheet.Rows.Length; i++)
            {
                string file = i < sheet.Opens.Length ? sheet.Opens[i] : "";
                list.Children.Add(Row(sheet.Rows[i], file.Length == 0 ? null : file));
            }
        }
        UIElement Row(string text, string? file)
        {
            if (file == null)
                return new TextBlock { Text = text, FontFamily = new FontFamily("Consolas"), Margin = new Thickness(0, 2, 0, 2) };
            var button = new Button
            {
                Content = text,
                HorizontalContentAlignment = HorizontalAlignment.Left,
                ToolTip = "打開這張不合格圖",
                Margin = new Thickness(0, 2, 0, 2),
                Tag = file,
            };
            button.Click += (_, _) => openNg(file);
            return button;
        }
        toggle.Click += (_, _) =>
        {
            onlyNg = !onlyNg;
            Reload();
        };
        Reload();
        var actions = new WrapPanel { Margin = new Thickness(0, 12, 0, 0) };
        actions.Children.Add(toggle);
        Show(owner, "產線明細", "", new DockPanel(), window =>
        {
            var root = new DockPanel { Margin = new Thickness(16) };
            DockPanel.SetDock(actions, Dock.Bottom);
            var scroll = new ScrollViewer { Content = list, VerticalScrollBarVisibility = ScrollBarVisibility.Auto };
            root.Children.Add(actions);
            var head = new StackPanel();
            head.Children.Add(new TextBlock { Text = "產線明細", FontSize = 16, FontWeight = FontWeights.Bold });
            head.Children.Add(note);
            DockPanel.SetDock(head, Dock.Top);
            root.Children.Add(head);
            root.Children.Add(scroll);
            window.Content = root;
        });
    }

    static void AddStat(Grid grid, int row, string label, string value)
    {
        while (grid.RowDefinitions.Count <= row)
            grid.RowDefinitions.Add(new RowDefinition());
        var name = new TextBlock { Text = label, Margin = new Thickness(0, 2, 12, 2), Foreground = new SolidColorBrush(Color.FromRgb(0x8B, 0x98, 0xAB)) };
        var number = new TextBlock { Text = value, Margin = new Thickness(0, 2, 0, 2) };
        Grid.SetRow(name, row);
        Grid.SetRow(number, row);
        Grid.SetColumn(number, 1);
        grid.Children.Add(name);
        grid.Children.Add(number);
    }

    static void Show(Window owner, string title, string note, UIElement body, Action<Window>? fill)
    {
        var window = new Window
        {
            Title = title,
            Width = 560,
            Height = 420,
            MinWidth = 420,
            MinHeight = 280,
            Owner = owner,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
            Background = new SolidColorBrush(Color.FromRgb(0x14, 0x17, 0x1C)),
            Foreground = new SolidColorBrush(Color.FromRgb(0xD5, 0xDE, 0xEA)),
        };
        if (fill != null)
        {
            fill(window);
            window.Show();
            return;
        }
        var close = new Button { Content = "關閉", HorizontalAlignment = HorizontalAlignment.Right, Margin = new Thickness(0, 12, 0, 0) };
        close.Click += (_, _) => window.Close();
        var root = new DockPanel { Margin = new Thickness(16) };
        DockPanel.SetDock(close, Dock.Bottom);
        var head = new StackPanel();
        head.Children.Add(new TextBlock { Text = title, FontSize = 16, FontWeight = FontWeights.Bold });
        if (note.Length > 0)
            head.Children.Add(new TextBlock { Text = note, Foreground = new SolidColorBrush(Color.FromRgb(0x8B, 0x98, 0xAB)), TextWrapping = TextWrapping.Wrap });
        DockPanel.SetDock(head, Dock.Top);
        root.Children.Add(close);
        root.Children.Add(head);
        root.Children.Add(body);
        window.Content = root;
        window.Show();
    }
}
