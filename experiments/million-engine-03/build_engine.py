"""Build the local batch scorer against the pinned PyMatching 2.4.0 source archive."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = "https://files.pythonhosted.org/packages/7d/19/39472c89d532e576890320eff5e9738e4b4d7828d027315f7e198ba0be8c/pymatching-2.4.0.tar.gz"
SHA = "0c30dafa506b086f50d15530debd7b7b39228597d22c52d5ccd114bcbb0a2326"


def main():
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    archive = build / "pymatching-2.4.0.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA:
        raise ValueError("Unexpected PyMatching source digest")
    with tarfile.open(archive) as source:
        source.extractall(build, filter="data")
    src = build / "pymatching-2.4.0" / "src"
    core = src / "pymatching" / "sparse_blossom"
    files = [ROOT / "engine.cpp"]
    for name in ("flooder", "flooder_matcher_interop", "matcher", "search", "tracker"):
        files.extend(p for p in sorted((core / name).glob("*.cc")) if p.name.count(".") == 1)
    if sys.platform == "win32":
        vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
        vs = subprocess.check_output([vswhere, "-latest", "-products", "*", "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
        setup = Path(vs) / "VC/Auxiliary/Build/vcvars64.bat"
        cmd = ["cl", "/nologo", "/std:c++20", "/EHsc", "/O2", "/DNDEBUG", "/MD", "/LD", "/MP4",
               f"/I{src}", *files, "/link", "/OUT:bath_engine.dll"]
        batch = build / "compile.cmd"
        batch.write_text(f'@call "{setup}" >nul\n@if errorlevel 1 exit /b 1\n' + subprocess.list2cmdline(cmd) + "\n")
        subprocess.run(["cmd", "/c", "compile.cmd"], cwd=build, check=True)
    else:
        subprocess.run([os.getenv("CXX", "c++"), "-std=c++20", "-O3", "-DNDEBUG", "-fPIC", "-shared",
                        f"-I{src}", *files, "-o", build / "bath_engine.so"], check=True)


if __name__ == "__main__":
    main()
