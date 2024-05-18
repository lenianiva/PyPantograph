#!/usr/bin/env python3

import subprocess
from pathlib import Path
from pantograph.server import Server

def get_lean_path():
    cwd = Path(__file__).parent.resolve() / 'Example'
    p = subprocess.check_output(['lake', 'env', 'printenv', 'LEAN_PATH'], cwd=cwd)
    return p

if __name__ == '__main__':
    lean_path = get_lean_path()
    print(lean_path)
