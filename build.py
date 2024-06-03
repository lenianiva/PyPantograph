#!/usr/bin/env python3

import subprocess, shutil, os, stat
from pathlib import Path

# -- Install Panograph
# Define paths for Pantograph source and Pantograph Python interface
PATH_PANTOGRAPH = Path("./src")
PATH_PY = Path("./pantograph")
# Run the `make` command in the PATH_PANTOGRAPH directory to build the Pantograph executable
with subprocess.Popen(["make"], cwd=PATH_PANTOGRAPH) as p:
    p.wait()
# Define the path to the executable
path_executable = PATH_PY / "pantograph"
# Copy the built Pantograph executable to the specified path
shutil.copyfile(PATH_PANTOGRAPH / ".lake/build/bin/pantograph", path_executable)
# Change the permissions of the Pantograph executable to make it executable
os.chmod(path_executable, os.stat(path_executable).st_mode | stat.S_IEXEC)

# -- Copy the Lean toolchain file to the specified path
shutil.copyfile(PATH_PANTOGRAPH / "lean-toolchain", PATH_PY / "lean-toolchain")