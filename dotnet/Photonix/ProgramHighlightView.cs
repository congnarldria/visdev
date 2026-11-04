using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace Photonix;

/// <summary>畫在程式文字後面，顏色和網頁版的語法高亮相同。位置跟文字方塊的每一行對齊。</summary>
sealed class ProgramHighlightView : FrameworkElement
{
    static readonly Brush Plain = Freeze(Color.FromRgb(0xE6, 0xED, 0xF3));
    static readonly Brush Comment = Freeze(Color.FromRgb(0x7F, 0x8B, 0x78));
    static readonly Brush String = Freeze(Color.FromRgb(0xE6, 0xC5, 0x6A));
    static readonly Brush Number = Freeze(Color.FromRgb(0xD2, 0xA4, 0xFF));
    static readonly Brush Key = Freeze(Color.FromRgb(0x8E, 0xB7, 0xFF));
    static readonly Brush Operator = Freeze(Color.FromRgb(0x7E, 0xE0, 0xA8));

    TextBox? Editor;
    HashSet<string> Operators = new(StringComparer.Ordinal);

    public void Attach(TextBox editor, IEnumerable<string> operators)
    {
        Editor = editor;
        Operators = new HashSet<string>(operators.Select(name => name.ToLowerInvariant()), StringComparer.Ordinal);
        editor.TextChanged += (_, _) => InvalidateVisual();
        editor.SizeChanged += (_, _) => InvalidateVisual();
        editor.Loaded += (_, _) => HookScroll();
        if (editor.IsLoaded)
            HookScroll();
        InvalidateVisual();
    }

    void HookScroll()
    {
        if (Editor?.Template.FindName("PART_ContentHost", Editor) is System.Windows.Controls.ScrollViewer scroll)
            scroll.ScrollChanged += (_, _) => InvalidateVisual();
    }

    protected override void OnRender(DrawingContext drawingContext)
    {
        if (Editor == null)
            return;
        string text = Editor.Text ?? "";
        double dpi = VisualTreeHelper.GetDpi(this).PixelsPerDip;
        var typeface = new Typeface(Editor.FontFamily, FontStyles.Normal, FontWeights.Normal, FontStretches.Normal);
        int index = 0;
        while (index <= text.Length)
        {
            int newline = text.IndexOf('\n', index);
            int end = newline < 0 ? text.Length : newline;
            string line = text[index..end].TrimEnd('\r');
            Rect rect = Editor.GetRectFromCharacterIndex(Math.Min(index, text.Length));
            double x = rect.X;
            double y = rect.Y;
            foreach (ProgramToken token in ProgramSyntax.Tokens(line, Operators))
            {
                if (token.Text.Length == 0)
                    continue;
                var formatted = new FormattedText(
                    token.Text,
                    CultureInfo.CurrentUICulture,
                    FlowDirection.LeftToRight,
                    typeface,
                    Editor.FontSize,
                    Brush(token.Kind),
                    dpi);
                drawingContext.DrawText(formatted, new Point(x, y));
                x += formatted.WidthIncludingTrailingWhitespace;
            }
            if (newline < 0)
                break;
            index = newline + 1;
        }
    }

    static Brush Brush(ProgramTokenKind kind) => kind switch
    {
        ProgramTokenKind.Comment => Comment,
        ProgramTokenKind.String => String,
        ProgramTokenKind.Number => Number,
        ProgramTokenKind.Key => Key,
        ProgramTokenKind.Operator => Operator,
        _ => Plain,
    };

    static Brush Freeze(Color color)
    {
        var brush = new SolidColorBrush(color);
        brush.Freeze();
        return brush;
    }
}
