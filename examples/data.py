#!/usr/bin/env python3

import subprocess
from pathlib import Path
from pantograph.server import Server

def get_project_and_lean_path():
    cwd = Path(__file__).parent.resolve() / 'Example'
    p = subprocess.check_output(['lake', 'env', 'printenv', 'LEAN_PATH'], cwd=cwd)
    return cwd, p

if __name__ == '__main__':
    project_path, lean_path = get_project_and_lean_path()
    print(f"$PWD: {project_path}")
    print(f"$LEAN_PATH: {lean_path}")
    server = Server(imports=['Example'], project_path=project_path, lean_path=lean_path)
    units, invocations = server.compile_unit("Example")
    for i, u in enumerate(units):
        print(f"==== #{i} ====")
        print(u)
    print("==== Invocations ====")
    for i in invocations:
        print(f"{i.before}\n{i.tactic}\n{i.after}\n")
