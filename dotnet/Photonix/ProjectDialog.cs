using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;

namespace Photonix;

sealed class ProjectDialog : Window
{
    readonly string Repo;
    readonly ListBox ProjectList = new() { MinHeight = 120 };
    readonly ListBox ProgramList = new() { MinHeight = 120 };
    readonly TextBox NameBox = new() { Width = 180, Margin = new Thickness(0, 0, 8, 0) };
    string? Current;

    public Func<bool> ConfirmDiscard { get; set; } = () => true;
    public Func<string> Source { get; set; } = () => "";
    public Func<string> ProgramName { get; set; } = () => "program.viz";
    public Action<string>? Noted { get; set; }
    public Action<string, string>? Opened { get; set; }
    public Action<string, string, string, IReadOnlyList<string>>? Saved { get; set; }
    public Action<string, string>? MakeProgram { get; set; }
    public Action<string, string>? Renamed { get; set; }

    public string? ChosenProject => Current;

    public ProjectDialog(string repo, string? current)
    {
        Repo = repo;
        Current = current;
        Title = "專案";
        Width = 560;
        SizeToContent = SizeToContent.Height;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        Background = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(0x14, 0x17, 0x1C));
        Foreground = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(0xE6, 0xED, 0xF3));
        var muted = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(0x8B, 0x94, 0x9E));

        var root = new StackPanel { Margin = new Thickness(16) };
        root.Children.Add(new TextBlock
        {
            Text = "一個專案放多支程式。圖可以放在專案資料夾，程式裡寫檔名就讀得到；samples/ 和 workspace/ 也還找得到。",
            TextWrapping = TextWrapping.Wrap,
            Foreground = muted,
            Margin = new Thickness(0, 0, 0, 10),
        });
        var create = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 0, 0, 8) };
        create.Children.Add(NameBox);
        var createButton = new Button { Content = "新專案" };
        createButton.Click += (_, _) => CreateProject();
        create.Children.Add(createButton);
        root.Children.Add(create);
        root.Children.Add(new TextBlock { Text = "專案", Margin = new Thickness(0, 4, 0, 2) });
        ProjectList.SelectionChanged += (_, _) =>
        {
            if (ProjectList.SelectedItem is string name)
            {
                Current = name;
                ReloadPrograms();
            }
        };
        root.Children.Add(ProjectList);
        root.Children.Add(new TextBlock { Text = "程式", Margin = new Thickness(0, 8, 0, 2) });
        ProgramList.MouseDoubleClick += (_, _) => OpenSelected();
        root.Children.Add(ProgramList);
        var actions = new WrapPanel { Margin = new Thickness(0, 12, 0, 0) };
        actions.Children.Add(Button("改名", (_, _) => RenameProject()));
        actions.Children.Add(Button("打開資料夾", (_, _) => OpenFolder()));
        actions.Children.Add(Button("把這支程式存進專案", (_, _) => SaveCurrent()));
        actions.Children.Add(Button("在專案裡新增程式", (_, _) => AddProgram()));
        actions.Children.Add(Button("關閉", (_, _) => Close()));
        root.Children.Add(actions);
        Content = root;
        Loaded += (_, _) => Reload();
    }

    static Button Button(string text, RoutedEventHandler click)
    {
        var button = new Button { Content = text, Margin = new Thickness(0, 0, 8, 8) };
        button.Click += click;
        return button;
    }

    void Note(string text) => Noted?.Invoke(text);

    void Reload()
    {
        try
        {
            ProjectList.ItemsSource = ProjectStore.List(Repo);
            ProjectList.SelectedItem = Current;
            ReloadPrograms();
        }
        catch (Exception ex)
        {
            Note(ex.Message);
        }
    }

    void ReloadPrograms()
    {
        if (string.IsNullOrEmpty(Current))
        {
            ProgramList.ItemsSource = new[] { "選一個專案，或先建立。" };
            return;
        }
        try
        {
            var programs = ProjectStore.Programs(Repo, Current);
            ProgramList.ItemsSource = programs.Count == 0 ? new[] { "這個專案還沒有程式。" } : programs;
        }
        catch (Exception ex)
        {
            Note(ex.Message);
        }
    }

    void CreateProject()
    {
        try
        {
            Current = ProjectStore.Create(Repo, NameBox.Text);
            NameBox.Text = "";
            Reload();
            Note("專案 " + Current);
        }
        catch (Exception ex)
        {
            Note(ex.Message);
        }
    }

    void RenameProject()
    {
        if (string.IsNullOrEmpty(Current))
        {
            Note("先選一個專案");
            return;
        }
        string? next = Prompt("新的專案名稱", Current);
        if (string.IsNullOrWhiteSpace(next) || next.Trim() == Current)
            return;
        try
        {
            string previous = Current;
            string renamed = ProjectStore.Rename(Repo, previous, next.Trim());
            Current = renamed;
            Renamed?.Invoke(previous, renamed);
            Reload();
            Note("專案 " + renamed);
        }
        catch (Exception ex)
        {
            Note(ex.Message);
        }
    }

    void OpenFolder()
    {
        if (string.IsNullOrEmpty(Current))
        {
            Note("先選一個專案");
            return;
        }
        try
        {
            string folder = ProjectStore.Folder(Repo, Current);
            bool opened = false;
            try
            {
                Process.Start(new ProcessStartInfo { FileName = folder, UseShellExecute = true });
                opened = true;
            }
            catch (Exception)
            {
                opened = false;
            }
            Note(opened ? "已打開資料夾 " + folder : "專案資料夾：" + folder);
        }
        catch (Exception ex)
        {
            Note(ex.Message);
        }
    }

    void SaveCurrent()
    {
        if (string.IsNullOrEmpty(Current))
        {
            Note("先建立或選一個專案");
            return;
        }
        try
        {
            ProjectProgram saved = ProjectStore.Save(Repo, Current, ProgramName(), Source());
            Saved?.Invoke(saved.Project, saved.Path, saved.Source, saved.Images);
            ReloadPrograms();
            Note("專案 " + saved.Project + " · " + saved.Name);
        }
        catch (Exception ex)
        {
            Note(ex.Message);
        }
    }

    void AddProgram()
    {
        if (string.IsNullOrEmpty(Current))
        {
            Note("先建立或選一個專案");
            return;
        }
        if (!ConfirmDiscard())
            return;
        try
        {
            var used = ProjectStore.Programs(Repo, Current).ToHashSet(StringComparer.OrdinalIgnoreCase);
            string filename = "program.viz";
            int number = 2;
            while (used.Contains(filename))
            {
                filename = "program" + number + ".viz";
                number++;
            }
            MakeProgram?.Invoke(Current, filename);
            ReloadPrograms();
        }
        catch (Exception ex)
        {
            Note(ex.Message);
        }
    }

    void OpenSelected()
    {
        if (ProgramList.SelectedItem is not string name || !name.EndsWith(".viz", StringComparison.OrdinalIgnoreCase) && !name.EndsWith(".hdv", StringComparison.OrdinalIgnoreCase))
            return;
        if (string.IsNullOrEmpty(Current) || !ConfirmDiscard())
            return;
        Opened?.Invoke(Current, name);
        Close();
    }

    string? Prompt(string title, string initial)
    {
        var box = new TextBox { Text = initial, Margin = new Thickness(0, 8, 0, 8), MinWidth = 240 };
        var ok = new Button { Content = "確定", IsDefault = true, MinWidth = 72, Margin = new Thickness(0, 0, 8, 0) };
        var cancel = new Button { Content = "取消", IsCancel = true, MinWidth = 72 };
        string? result = null;
        var buttons = new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right };
        buttons.Children.Add(ok);
        buttons.Children.Add(cancel);
        var panel = new StackPanel { Margin = new Thickness(16) };
        panel.Children.Add(new TextBlock { Text = title });
        panel.Children.Add(box);
        panel.Children.Add(buttons);
        var prompt = new Window
        {
            Title = "專案",
            Owner = this,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
            SizeToContent = SizeToContent.WidthAndHeight,
            Content = panel,
            Background = Background,
            Foreground = Foreground,
        };
        ok.Click += (_, _) =>
        {
            result = box.Text;
            prompt.DialogResult = true;
        };
        prompt.ShowDialog();
        return result;
    }
}
