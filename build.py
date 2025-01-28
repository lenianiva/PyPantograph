#!/usr/bin/env python3

import subprocess, shutil, os, stat
from pathlib import Path

# -- Install Panograph
# Define paths for Pantograph source and Pantograph Python interface
PATH_PANTOGRAPH = Path("./src")
PATH_PY = Path("./pantograph")
with subprocess.Popen(["lake", "build", "repl"], cwd=PATH_PANTOGRAPH) as p:
    p.wait()

path_executable = PATH_PY / "pantograph-repl"
if os.name == 'nt': # check that the OS is Windows or not
    shutil.copyfile(PATH_PANTOGRAPH / ".lake/build/bin/repl.exe", path_executable) 
else:
    shutil.copyfile(PATH_PANTOGRAPH / ".lake/build/bin/repl", path_executable)
os.chmod(path_executable, os.stat(path_executable).st_mode | stat.S_IEXEC)

# -- Copy the Lean toolchain file to the specified path
shutil.copyfile(PATH_PANTOGRAPH / "lean-toolchain", PATH_PY / "lean-toolchain")
