# Anima Studio - PyInstaller build spec.
#
# Run this ON WINDOWS (PyInstaller builds for whatever OS it runs on -
# it can't cross-compile a .exe from Linux/Mac):
#
#     pip install -r requirements.txt
#     pyinstaller anima_studio.spec --noconfirm
#
# Or just double-click build_windows.bat, which does the same thing.
# Output: dist\Anima Studio\Anima Studio.exe
#
# This is "onedir" mode (a folder containing the exe + its files) rather
# than "onefile" - onefile re-extracts itself to a temp folder on every
# launch, which makes a Flask+webview app noticeably slower to open.
# The whole dist\Anima Studio\ folder is what you'd zip up to share or
# ship - not just the .exe alone.

import os

# native_backend (the trainer server + its requirements.txt, which lists peft/diffusers)
# is NOT imported by the app itself, so PyInstaller would miss it - it must be listed here.
# training_backend is bundled WITHOUT its venv (multi-GB: torch + CUDA) and
# without caches - that is what made the build take forever. The backend's
# dependencies are installed on the user's machine from the app (Backend tab).
_SKIP_DIRS = {"venv", ".venv", "__pycache__", ".git", ".pytest_cache"}


def _tree(src, dst):
    out = []
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        rel = os.path.relpath(root, src)
        target = dst if rel == "." else os.path.join(dst, rel)
        for f in files:
            if f.endswith((".pyc", ".pyo")):
                continue
            out.append((os.path.join(root, f), target))
    return out


datas = [
    ("app/templates", "app/templates"),
    ("app/static", "app/static"),
] + _tree("training_backend", "training_backend") + _tree("native_backend", "native_backend")

# pywebview picks its Windows backend (WebView2, or the older winforms/
# MSHTML fallback) at runtime, which PyInstaller's static import scan
# can miss - listed explicitly here so the built exe doesn't fail with
# a "no module named webview.platforms.X" error on launch.
hiddenimports = [
    "webview.platforms.winforms",
    "webview.platforms.edgechromium",
]

a = Analysis(
    ["desktop.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Anima Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon="icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Anima Studio",
)
