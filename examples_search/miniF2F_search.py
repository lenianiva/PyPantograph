#!/usr/bin/env python3

import subprocess, json
from pathlib import Path
from pantograph.server import Server
from pantograph.search_llm import LLMAgent

def get_project_and_lean_path():
    cwd = Path(__file__).parent.resolve() / 'Example'
    p = subprocess.check_output(['lake', 'env', 'printenv', 'LEAN_PATH'], cwd=cwd)
    return cwd, p

def read_test_data():
    jsonl_path = Path(__file__).parent / 'test.jsonl'
    with open(jsonl_path, 'r') as f:
        return [json.loads(l) for l in list(f)]

def try_test_data(server, agent, entry) -> bool:
    e = entry["formal_statement"]
    informal_stmt = entry["informal_stmt"]
    informal_proof = entry["informal_proof"]
    key_theorem, name, e = e.split(' ', 2)
    e, tail = e.split(':=', 1)
    target = "forall " + ','.join(e.rsplit(':', 1))
    print(f"Target: {target}")
    agent = LLMAgent(server)
    return agent.search(server=server, target=target, informal_stmt = informal_stmt, informal_proof = informal_proof,verbose=True)

if __name__ == '__main__':
    project_path, lean_path = get_project_and_lean_path()
    print(f"$PWD: {project_path}")
    print(f"$LEAN_PATH: {lean_path}")

    test_data = read_test_data()
    server = Server(imports=["Mathlib"], project_path=project_path, lean_path=lean_path)
    agent = LLMAgent(server)
    try_test_data(server, agent, test_data[0])
