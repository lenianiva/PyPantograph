#!/usr/bin/env python3

import subprocess, json, argparse
from typing import Optional
from pathlib import Path
from pantograph.server import Server
from pantograph.search import SearchResult
from pantograph.search_llm import LLMAgent

def get_project_and_lean_path():
    cwd = Path(__file__).parent.resolve() / 'Example'
    p = subprocess.check_output(['lake', 'env', 'printenv', 'LEAN_PATH'], cwd=cwd)
    return cwd, p

def read_test_data(use_valid: bool):
    jsonl_path = Path(__file__).parent / ('valid.jsonl' if use_valid else 'test.jsonl')
    with open(jsonl_path, 'r') as f:
        return [json.loads(l) for l in list(f)]

def try_test_data(server, agent, entry: dict, max_steps: int) -> Optional[SearchResult]:
    e = entry["formal_statement"]
    informal_stmt = entry["informal_stmt"]
    informal_proof = entry["informal_proof"]

    key_position = e.find('theorem')
    if key_position == -1:
        # Can't output anything for this one
        return None
    e = e[key_position:]
    key_theorem, name, e = e.split(' ', 2)
    e, tail = e.split(':=', 1)
    target = "forall " + ','.join(e.rsplit(':', 1))
    print(f"Target: {target}")
    agent = LLMAgent(server)
    return agent.search(server=server, target=target, informal_stmt = informal_stmt, informal_proof = informal_proof,verbose=True, max_steps=max_steps)

def output_file_name(datum, use_hammer: bool, use_llm: bool):
    name = datum["id"]
    folder = 'output'
    if use_hammer:
        folder += '-hammer'
    if use_llm:
        folder += '-llm'
    folder = Path(__file__).parent / folder
    folder.mkdir(exist_ok=True, parents=True)
    return folder / f"{name}.json"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='MiniF2F Search',
                    description='Executes LLM on MiniF2F Search')
    parser.add_argument('--use-hammer', action='store_true')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--use-llm', action='store_true')
    parser.add_argument('-s', '--max-steps', default=200)
    args = parser.parse_args()

    project_path, lean_path = get_project_and_lean_path()
    print(f"$PWD: {project_path}")
    print(f"$LEAN_PATH: {lean_path}")

    test_data = read_test_data(args.validation)
    server = Server(imports=["Mathlib"], project_path=project_path, lean_path=lean_path)
    agent = LLMAgent(server, use_hammer=args.use_hammer, use_llm=args.use_llm)
    for datum in test_data:
        file_name = output_file_name(datum, args.use_hammer, args.use_llm)
        if file_name.is_file():
            print(f"Skipping {datum['id']}")
            continue
        result = try_test_data(server, agent, datum, max_steps=args.max_steps)
        if result is None:
            with open(file_name + '-placeholder', 'w') as f:
                json.dump({ 'id': datum['id'] }, f)
        else:
            with open(file_name, 'w') as f:
                json.dump({ 'id': datum['id'], 'success': result.success, 'steps': result.steps  }, f)
