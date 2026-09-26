"""
Builds the portable NUMBSKULL download: one zip with its own private copy of
Python, every library the AI needs, the pretrained model and the panel.
Whoever downloads it extracts it and double-clicks NUMBSKULL.exe -- no Python
install, no pip.

USAGE (on Windows, with the same Python version the zip should contain):
    python build_portable.py

Output: dist/NUMBSKULL-v<version>-portable.zip (dist/ is ignored by git).
Inside: a NUMBSKULL folder with just NUMBSKULL.exe (the launcher, built from
launcher/NUMBSKULL.cs with the C# compiler that comes with Windows),
READ ME FIRST.txt, and a "files" folder with everything else.

How it's put together:
  - Python itself is the official "embeddable" build from python.org for
    this exact version (signed by the Python Software Foundation, so
    antivirus programs trust it more than a packed .exe).
  - The embeddable build leaves out tkinter, which the panel needs, so
    tkinter and its Tcl/Tk files are copied from this machine's install.
  - Libraries from requirements.txt are pip-installed into the bundle.
  - The project's files come from git (committed files only), plus
    model.joblib and ball_classifier.joblib if they're here -- never your
    recordings, logs or panel settings.
"""

import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import app  # for VERSION

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"
NAME = f"NUMBSKULL-v{app.VERSION}-portable"  # the zip's name
FOLDER = "NUMBSKULL"  # the folder inside it: short, see the path check in the launcher
BUILD = DIST / FOLDER
FILES = BUILD / "files"  # everything but NUMBSKULL.exe and the READ ME
PY = FILES / "python"
CSC = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")  # comes with Windows

README = f'''NUMBSKULL v{app.VERSION} -- portable
=================================

1. Extract the NUMBSKULL folder somewhere with a short path, e.g. straight
   into Documents (not inside the zip, and not buried in other folders).
2. Double-click NUMBSKULL.exe.
   If Windows shows "Windows protected your PC": More info -> Run anyway.
3. In the panel, press Start AI.

Nothing to install: Python and everything else are inside the "files"
folder. Keep NUMBSKULL.exe next to it.

The pretrained model is included. It plays best with the setup it was
trained on: 1920x1080 screen, Roblox windowed and maximized, camera zoom
12 notches out from first person, shift lock on.

Full guide: {app.REPO_URL}

This is an educational project. Using a bot in Roblox games is against
Roblox's Terms of Use and can get your account banned.
'''


def run(*argv):
    print(">", " ".join(str(a) for a in argv))
    subprocess.run([str(a) for a in argv], check=True)


def main():
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    if BUILD.exists():
        shutil.rmtree(BUILD)
    PY.mkdir(parents=True)

    # 1. Embeddable Python, same version as this one.
    url = f"https://www.python.org/ftp/python/{version}/python-{version}-embed-amd64.zip"
    print("Downloading", url)
    with urllib.request.urlopen(url) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(PY)

    # 2. tkinter (left out of the embeddable build), from this install.
    base = Path(sys.base_prefix)
    shutil.copytree(base / "Lib" / "tkinter", PY / "Lib" / "tkinter",
                    ignore=shutil.ignore_patterns("__pycache__", "test"))
    shutil.copytree(base / "tcl", PY / "tcl",
                    ignore=shutil.ignore_patterns("*.lib", "*.sh", "nmake"))
    for dll in (base / "DLLs").glob("*"):
        if dll.name == "_tkinter.pyd" or dll.name.startswith(("tcl", "tk", "zlib")) \
                and dll.suffix == ".dll":
            shutil.copy2(dll, PY)

    # 3. Search path: the bundled libraries and the app folder (one up).
    pth = next(PY.glob("python*._pth"))
    pth.write_text(f"{pth.stem}.zip\n.\nLib\nLib\\site-packages\n..\nimport site\n")

    # 4. Libraries.
    run(sys.executable, "-m", "pip", "install", "--no-warn-script-location", "--disable-pip-version-check",
        "--target", PY / "Lib" / "site-packages", "-r", HERE / "requirements.txt")

    # Test suites and build leftovers aren't needed to run: smaller zip,
    # shorter paths.
    site = PY / "Lib" / "site-packages"
    for d in [d for d in site.rglob("tests") if d.is_dir()]:
        shutil.rmtree(d, ignore_errors=True)
    for pattern in ("*.lib", "*.pxd", "*.pyx", "*.tp", "*.pyi"):
        for f in site.rglob(pattern):
            f.unlink()

    # 5. The project (committed files) and the pretrained model.
    archive = subprocess.run(["git", "archive", "HEAD"], cwd=HERE, capture_output=True, check=True).stdout
    tar_path = DIST / "_src.tar"
    tar_path.write_bytes(archive)
    shutil.unpack_archive(tar_path, FILES)
    tar_path.unlink()
    for f in ("Blade Ball AI.bat", "build_portable.py", ".gitignore", ".gitattributes"):
        (FILES / f).unlink(missing_ok=True)
    shutil.rmtree(FILES / "launcher", ignore_errors=True)
    for f in ("model.joblib", "ball_classifier.joblib"):
        if (HERE / f).exists():
            shutil.copy2(HERE / f, FILES)
        else:
            print(f"(no {f} here -- the download won't include a trained model)")
    (BUILD / "READ ME FIRST.txt").write_text(README, newline="\r\n")

    # 6. NUMBSKULL.exe (launcher/NUMBSKULL.cs), told how long the folder's
    # own path can be before the deepest file inside goes over Windows' 260
    # character limit.
    longest = max(len(str(f.relative_to(BUILD))) for f in BUILD.rglob("*"))
    source = (HERE / "launcher" / "NUMBSKULL.cs").read_text(encoding="utf-8")
    cs = DIST / "_NUMBSKULL.cs"
    cs.write_text(source.replace("__MAX_FOLDER_LENGTH__", str(259 - longest)), encoding="utf-8")
    run(CSC, "/nologo", "/target:winexe", "/optimize", f"/win32icon:{HERE / 'assets' / 'numbskull.ico'}",
        f"/out:{BUILD / 'NUMBSKULL.exe'}", cs)
    cs.unlink()

    # 7. Zip it.
    out = shutil.make_archive(str(DIST / NAME), "zip", DIST, FOLDER)
    print(f"Deepest file is {longest} characters inside the folder, so the folder's own path "
          f"can be up to {259 - longest}")
    size = Path(out).stat().st_size / 1e6
    print(f"\nBuilt {out} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
