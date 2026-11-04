using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace Photonix;

sealed class TemplateDialog : Window
{
    readonly ListBox List = new() { MinHeight = 280 };

    public Func<bool> Confirm { get; set; } = () => true;
    public Action<TemplateEntry>? Opened { get; set; }

    public TemplateDialog()
    {
        Title = "範本一覽";
        Width = 640;
        Height = 520;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        Background = Brush(0x14, 0x17, 0x1C);
        Foreground = Brush(0xE6, 0xED, 0xF3);
        var muted = Brush(0x8B, 0x94, 0x9E);

        var root = new DockPanel { Margin = new Thickness(16) };
        var close = new Button { Content = "關閉", MinWidth = 72, HorizontalAlignment = HorizontalAlignment.Right, Margin = new Thickness(0, 12, 0, 0) };
        close.Click += (_, _) => Close();
        DockPanel.SetDock(close, Dock.Bottom);
        root.Children.Add(close);
        root.Children.Add(new TextBlock
        {
            Text = "點一列就打開。右邊是跑完該看到的結果。",
            Foreground = muted,
            Margin = new Thickness(0, 0, 0, 10),
            TextWrapping = TextWrapping.Wrap,
        });
        List.Background = Brush(0x0E, 0x11, 0x16);
        List.Foreground = Foreground;
        List.BorderBrush = Brush(0x2D, 0x33, 0x3B);
        foreach (TemplateEntry entry in ProgramLibrary.Templates)
        {
            var row = new DockPanel { LastChildFill = true, Margin = new Thickness(2) };
            var title = new TextBlock
            {
                Text = entry.Title,
                FontWeight = FontWeights.Bold,
                Width = 96,
                VerticalAlignment = VerticalAlignment.Center,
            };
            DockPanel.SetDock(title, Dock.Left);
            row.Children.Add(title);
            row.Children.Add(new TextBlock
            {
                Text = entry.Expect,
                Foreground = muted,
                TextWrapping = TextWrapping.Wrap,
                Margin = new Thickness(8, 0, 0, 0),
                VerticalAlignment = VerticalAlignment.Center,
            });
            List.Items.Add(new ListBoxItem { Content = row, Tag = entry, Padding = new Thickness(4, 6, 4, 6) });
        }
        List.PreviewMouseLeftButtonUp += (_, eventArgs) =>
        {
            if (ItemFrom(eventArgs.OriginalSource) is TemplateEntry entry)
                Pick(entry);
        };
        root.Children.Add(List);
        Content = root;
    }

    void Pick(TemplateEntry entry)
    {
        if (!Confirm())
            return;
        Opened?.Invoke(entry);
        DialogResult = true;
    }

    TemplateEntry? ItemFrom(object? source)
    {
        if (source is not DependencyObject node)
            return null;
        var item = ItemsControl.ContainerFromElement(List, node) as ListBoxItem;
        return item?.Tag as TemplateEntry;
    }

    static SolidColorBrush Brush(byte r, byte g, byte b) => new(Color.FromRgb(r, g, b));
}
