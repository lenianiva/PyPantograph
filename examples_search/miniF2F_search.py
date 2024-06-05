#!/usr/bin/env python3

import subprocess
from pathlib import Path
from pantograph.server import Server
from pantograph.search_llm import LLMAgent

def get_project_and_lean_path():
    cwd = Path(__file__).parent.resolve() / 'Example'
    p = subprocess.check_output(['lake', 'env', 'printenv', 'LEAN_PATH'], cwd=cwd)
    return cwd, p

if __name__ == '__main__':
    project_path, lean_path = get_project_and_lean_path()
    print(f"$PWD: {project_path}")
    print(f"$LEAN_PATH: {lean_path}")

    server = Server(imports=["Mathlib"], project_path=project_path, lean_path=lean_path)
    target = "∀ (b h v : ℝ)  (h₀ : 0 < b ∧ 0 < h ∧ 0 < v)  (h₁ : v = 1 / 3 * (b * h))  (h₂ : b = 30)  (h₃ : h = 13 / 2) , v = 65"
    # target = "theorem mathd_algebra_478\n  (b h v : ℝ)\n  (h₀ : 0 < b ∧ 0 < h ∧ 0 < v)\n  (h₁ : v = 1 / 3 * (b * h))\n  (h₂ : b = 30)\n  (h₃ : h = 13 / 2) :\n  v = 65 := sorry"
    agent = LLMAgent(server)
    flag = agent.search(server=server, target=target, verbose=True)

