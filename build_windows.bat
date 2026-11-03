@echo off
cd /d "%~dp0"
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if errorlevel 1 (
  echo 找不到可用的 Python 3。
  echo 請安裝 Python 3.9 或更新版本，並勾選 Add python.exe to PATH。
  echo 若已經安裝，請關掉命令提示字元再打開，然後重跑這個檔。
  exit /b 1
)
if not exist "app\engine\visdev_match.pyd" (
  echo 找不到 app\engine\visdev_match.pyd
  echo 這是 Windows 的形狀匹配檔，要留在專案裡才能打包。
  echo 請確認這個檔還在。Linux 的 visdev_match.so 不能拿來打包。
  exit /b 1
)
python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
  echo 找不到 PyInstaller。請先執行：
  echo   python -m pip install pyinstaller
  exit /b 1
)
python -m PyInstaller --noconfirm --windowed --name VisDev --paths . --hidden-import app.engine.visdev_match --add-binary "app\engine\visdev_match.pyd;app\engine" --add-data "app\static;app\static" --add-data "samples;samples" app\desktop.py
if errorlevel 1 (
  echo 打包失敗。原因在上方輸出。
  echo 常見原因是 PyInstaller 沒裝好，或 app\engine\visdev_match.pyd 無法被這個 Python 讀取。
  exit /b 1
)
if not exist "dist\VisDev\VisDev.exe" (
  echo 打包指令跑完了，但是沒有看到 dist\VisDev\VisDev.exe。
  exit /b 1
)
echo 執行檔：dist\VisDev\VisDev.exe
echo 若雙擊沒有視窗，請看專案目錄裡的 visdev.log。
