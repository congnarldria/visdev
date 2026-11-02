@echo off
cd /d "%~dp0"
if not exist "app\engine\visdev_match.pyd" (
  echo 找不到 app\engine\visdev_match.pyd
  echo 請確認 Windows 的原生匹配檔還在專案裡。
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
  exit /b 1
)
echo 執行檔：dist\VisDev\VisDev.exe
echo 若雙擊沒有視窗，請看專案目錄裡的 visdev.log。
