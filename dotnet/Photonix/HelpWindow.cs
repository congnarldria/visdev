using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace Photonix;

/// <summary>使用說明和關於。不擋住主視窗。</summary>
static class HelpWindow
{
    public static void ShowGuide(Window owner) => Show(owner, "使用說明", Guide());

    public static void ShowAbout(Window owner) => Show(owner, "關於 Photonix", About());

    static UIElement Guide()
    {
        var panel = new StackPanel();
        Add(panel, "Photonix 是 WPF 視覺開發環境。底層呼叫 Vizion.Operators，找到 C 核心就用，找不到就用 C#。", bold: true);
        Heading(panel, "執行");
        Add(panel, "F5 執行到斷點或結尾，F6 單步，F7 重設，也可以停止。錯誤停在那一行，已經算出的圖和變數留著。行號可設斷點，拖綠色箭頭改下一行。追蹤會把每一行寫進輸出。");
        Heading(panel, "版面");
        Add(panel, "程式、圖形視窗、變數、算子、語法、輸出、錯誤清單可以拖成浮動視窗，再拖回去。可自動隱藏、釘選。版面記在這台電腦。檢視選單可隱藏窗格，全部釘選一次釘回，回復版面回到預設。");
        Heading(panel, "圖形視窗");
        Add(panel, "滾輪以游標縮放，手抓移動，拉框放大。點選會讀出那一點，也可以點一個區域或輪廓。線寬 1 到 10。存畫面含底圖和疊加。算子參數窗開著時可畫線、框、圓、橢圓，確認後填進參數。");
        Add(panel, "二值化、次像素、特徵是獨立視窗，不擋住主視窗。二值化拖統計圖時區域畫在那個視窗，放開才改程式並重跑。次像素用八連通拆開，點一個物件畫白色虛線。特徵要先點一個區域或輪廓，再勾選要看的數值。");
        Add(panel, "OCR 可選引擎，把讀字那一行寫進程式並重跑。自動只用本機通用引擎，雲端引擎會把影像送到該服務。connection 用距離和面積下限，每個物件一種顏色。1 是八連通。");
        Heading(panel, "範本");
        Add(panel, "檔案、範本一覽列出每一支該有的結果，點一列就打開。");
        Bullet(panel, "零件檢測：samples/parts.png，8 個零件。");
        Bullet(panel, "瑕疵：samples/hard_contrast.png。暗塊 1 個、面積 3550；亮點 1 個、面積 144。圖形視窗可以改灰階範圍。");
        Bullet(panel, "色彩：samples/color_part.png。綠塊面積 1200、色調 60。超出預設範圍就不合格。");
        Bullet(panel, "字元：samples/ocr_label.png 寫著 VISDEV 168。圖形視窗可以改預期字，改過的字會記在這台電腦。");
        Bullet(panel, "卡尺：直徑約 83.93，寬 140，高 90。上下限記在這台電腦，按預設回到檔案裡的數字。");
        Bullet(panel, "條紋寬度：samples/stripes.png。最寬的亮條是 22。");
        Bullet(panel, "產線：samples/line_batch 有 5 張。合格 3、不合格 2。不合格圖在 workspace/line_ng，明細在 workspace/line_result.csv。這支用檔案資料夾，不開相機。");
        Bullet(panel, "產線相機：編號 0。Windows 先問 DirectShow，打不開再試 OpenCV。沒有相機時改跑檔案產線。");
        Bullet(panel, "產線輸出：每一張送一行 VISDEV,OK 或 NG,合格張數,不合格張數,檔名。圖形視窗可選 TCP、序列埠或 Modbus。預設是 TCP loopback。序列埠預設名稱是 loopback，不開真的 COM。Windows 開 COM 的方式沒有改。");
        Bullet(panel, "孔數：samples/holes.png 有 4 個暗孔。按設為預期把這次量到的圓心寫回，再按執行，偏差是 0。");
        Bullet(panel, "孔數產線：samples/hole_batch。合格 2、不合格 3。明細在 workspace/line_holes.csv。");
        Bullet(panel, "次像素邊緣：samples/inspect_edge.viz。階躍在行 15.5。");
        Heading(panel, "產線明細");
        Add(panel, "程式有明細時，狀態列可看不合格圖、統計和看明細。統計是整份 CSV 的總數、合格、不合格和良率。明細預設看最後 20 列，可以只看不合格。點不合格那一列會打開那張圖，再按一次回到程式畫面。不合格圖留在 workspace/line_ng。");
        Heading(panel, "存檔");
        Add(panel, "可以新增、開啟、儲存、另存。副檔名接受 viz、hdv、hdev 等。viz 以 CRLF 儲存。一個專案可以放多支程式。圖放在專案資料夾時，程式寫檔名就讀得到。samples/ 和 workspace/ 也找得到。");
        return panel;
    }

    static UIElement About()
    {
        var panel = new StackPanel();
        Add(panel, "Photonix", bold: true);
        Add(panel, "視覺開發環境。程式寫法接近 HDevelop，底層呼叫 Vizion.Operators。");
        Add(panel, "檔案可以新增、開啟、儲存、另存、載入影像，也可以把目前程式匯出成呼叫 Vizion.Operators 的 C#。編輯可以復原、重做、剪下、複製、貼上、全選。檢視可以隱藏窗格、全部釘選、回復版面。執行是 F5、F6、F7。運算子依分類列出，點程式裡的算子名稱可改參數。工具會開二值化、次像素和特徵。圖形列還有 OCR 和 connection。說明是這份使用說明。");
        return panel;
    }

    static void Heading(StackPanel panel, string text)
    {
        panel.Children.Add(new TextBlock
        {
            Text = text,
            FontWeight = FontWeights.Bold,
            Margin = new Thickness(0, 12, 0, 4),
        });
    }

    static void Add(StackPanel panel, string text, bool bold = false)
    {
        panel.Children.Add(new TextBlock
        {
            Text = text,
            TextWrapping = TextWrapping.Wrap,
            FontWeight = bold ? FontWeights.Bold : FontWeights.Normal,
            Margin = new Thickness(0, 0, 0, 6),
        });
    }

    static void Bullet(StackPanel panel, string text)
    {
        panel.Children.Add(new TextBlock
        {
            Text = "· " + text,
            TextWrapping = TextWrapping.Wrap,
            Margin = new Thickness(12, 0, 0, 4),
        });
    }

    static void Show(Window owner, string title, UIElement body)
    {
        var window = new Window
        {
            Title = title,
            Width = 680,
            Height = 560,
            MinWidth = 480,
            MinHeight = 320,
            Owner = owner,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
            Background = new SolidColorBrush(Color.FromRgb(0x14, 0x17, 0x1C)),
            Foreground = new SolidColorBrush(Color.FromRgb(0xD5, 0xDE, 0xEA)),
        };
        var close = new Button { Content = "確定", HorizontalAlignment = HorizontalAlignment.Right, Margin = new Thickness(0, 12, 0, 0), IsCancel = true };
        close.Click += (_, _) => window.Close();
        var root = new DockPanel { Margin = new Thickness(16) };
        DockPanel.SetDock(close, Dock.Bottom);
        var scroll = new ScrollViewer { Content = body, VerticalScrollBarVisibility = ScrollBarVisibility.Auto };
        root.Children.Add(close);
        root.Children.Add(scroll);
        window.Content = root;
        window.Show();
    }
}
