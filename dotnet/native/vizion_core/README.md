# vizion_core

閾值化的 C 核心。輸出 `vizion_core.dll`（Windows）或 `libvizion_core.so`（Linux）。x86 放 `bin/x86`，x64 放 `bin/x64`。C# 依行程位元選擇，找不到就用原來的 C#。

## Visual Studio

開啟 `dotnet/native/vizion_core/vizion_core.sln`。組態選 Release，平台選 **x64** 或 **x86**（x86 會建 Win32）。起始專案用 `vizion_core_check`，F5 印出「閾值自我檢查通過」。

兩個平台都要各建一次。DLL 會出現在 `bin/x64/vizion_core.dll` 或 `bin/x86/vizion_core.dll`。

純 C 在 `threshold_scalar.c`。Win32 這個檔用 `/arch:IA32`。SSE2、AVX2 各有一個檔，執行時再選。x86 的匯出是 `__cdecl`。
