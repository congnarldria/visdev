using System.Diagnostics;
using Vizion;

var pixels = new byte[8 * 8];
pixels[2 * 8 + 3] = 200;
var frame = ImageFrame.Gray(8, 8, pixels);
var dest = new byte[8 * 8 * 4];
FrameComposer.Render(frame, new ScreenMap(0, 0, 8, 8, 0, 0, 8, 8), Array.Empty<Overlay>(), dest, 8, 8);
int gray = dest[(2 * 8 + 3) * 4];
if (gray != 200)
    throw new InvalidOperationException($"1:1 像素應是 200，得到 {gray}");

const int fullWidth = 4096;
const int fullHeight = 3072;
var big = new byte[fullWidth * fullHeight];
for (int i = 0; i < big.Length; i += 64)
    big[i] = 180;
var bigFrame = ImageFrame.Gray(fullWidth, fullHeight, big);
var view = new byte[640 * 480 * 4];
var roi = new ScreenMap(0, 0, 640, 480, 100, 200, 100 + 480, 200 + 640);

var clock = Stopwatch.StartNew();
long sink = 0;
for (int n = 0; n < 20; n++)
{
    for (int i = 0; i < big.Length; i++)
        sink += big[i];
}
clock.Stop();
var fullMs = clock.Elapsed.TotalMilliseconds;
if (sink == 0)
    throw new InvalidOperationException("全圖掃描被省掉了");

clock.Restart();
for (int n = 0; n < 20; n++)
    FrameComposer.Render(bigFrame, roi, Array.Empty<Overlay>(), view, 640, 480);
clock.Stop();
var roiMs = clock.Elapsed.TotalMilliseconds;
if (!(roiMs * 3 < fullMs))
    throw new InvalidOperationException($"可見範圍 {roiMs:0.0} ms 沒有比整張掃描 {fullMs:0.0} ms 快過三倍");

var wide = new byte[20 * 5];
for (int i = 0; i < wide.Length; i++)
    wide[i] = 40;
var wideFrame = ImageFrame.Gray(20, 5, wide);
var fitted = new byte[80 * 20 * 4];
FrameComposer.Render(wideFrame, ScreenMap.Fit(20, 5, 80, 20), Array.Empty<Overlay>(), fitted, 80, 20);
if (fitted[0] == 0 && fitted[1] == 0 && fitted[2] == 0)
    throw new InvalidOperationException("縮小後的長條沒有畫出來");

var core = new VizionViewCore();
core.Resize(64, 64);
var tiny = ImageFrame.Gray(64, 64, new byte[64 * 64]);
for (int n = 0; n < 40; n++)
    core.Show(tiny);
var until = DateTime.UtcNow.AddSeconds(2);
while (DateTime.UtcNow < until && core.Presented + core.Coalesced < 40)
    Thread.Sleep(5);
if (core.Coalesced < 1)
    throw new InvalidOperationException("連續畫面沒有合併");
if (core.Presented + core.Coalesced < 40)
    throw new InvalidOperationException($"送出 {core.Presented} 張、合併 {core.Coalesced} 張，加起來不到 40");

Console.WriteLine($"整張掃描 20 次 {fullMs:0.0} ms");
Console.WriteLine($"可見 640x480 20 次 {roiMs:0.0} ms");
Console.WriteLine($"合併 {core.Coalesced} 張，實際畫了 {core.Presented} 張");
