#!/usr/bin/env python3

import subprocess, shutil, os, stat
from pathlib import Path

PATH_PANTOGRAPH = Path("./src")
PATH_PY = Path("./pantograph")

with subprocess.Popen(["make"], cwd=PATH_PANTOGRAPH) as p:
    p.wait()

path_executable = PATH_PY / "pantograph"
shutil.copyfile(PATH_PANTOGRAPH / ".lake/build/bin/pantograph", path_executable)
os.chmod(path_executable, os.stat(path_executable).st_mode | stat.S_IEXEC)
shutil.copyfile(PATH_PANTOGRAPH / "lean-toolchain", PATH_PY / "lean-toolchain")
