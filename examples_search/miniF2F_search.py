#!/usr/bin/env python3

import subprocess, json, argparse
from typing import Optional
from pathlib import Path
from pantograph.server import Server, ServerError
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

def inplace_to_statement(expr: str) -> str:
    bracket = 0
    i = 0
    while i < len(expr):
        if expr[i] == ':' and bracket == 0:
            break
        elif expr[i] == '(':
            bracket += 1
        elif expr[i] == ')':
            bracket -= 1
        i += 1
    if i == 0:
        return expr[1:]
    if i == len(expr):
        return expr

    return 'forall ' + expr[:i] + ' , ' + expr[i+1:]


def try_test_data(server, agent, entry: dict, max_steps: int, max_trials_per_goal: int) -> Optional[SearchResult]:
    e = entry["formal_statement"]
    print(e)
    informal_stmt = entry["informal_stmt"]
    informal_proof = entry["informal_proof"]

    key_position = e.find('theorem')
    if key_position != 0:
        # Can't output anything for this one
        return None
    e = e[key_position:]
    # remove the tail := sorry
    e, tail = e.rsplit(':=', 1)
    # remove the head
    key_theorem, name, e = e.split(' ', 2)
    target = inplace_to_statement(e.strip())
    print(f"Target: {target}")
    try:
        return agent.search(server=server, target=target, informal_stmt = informal_stmt, informal_proof = informal_proof,verbose=True,
                            max_steps=max_steps, max_trials_per_goal=max_trials_per_goal)
    except ServerError as e:
        return None

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
    parser.add_argument('-t', '--max-trials-per-goal', default=4)
    args = parser.parse_args()

    project_path, lean_path = get_project_and_lean_path()
    print(f"$PWD: {project_path}")
    print(f"$LEAN_PATH: {lean_path}")

    test_data = read_test_data(args.validation)
    for datum in test_data:
        file_name = output_file_name(datum, args.use_hammer, args.use_llm)
        placeholder_file_name = file_name.with_suffix('.placeholder')
        if file_name.is_file():
            print(f"Skipping {datum['id']}")
            continue
        server = Server(imports=["Example"], project_path=project_path, lean_path=lean_path, options=["maxHeartbeats=0"])
        agent = LLMAgent(server, use_hammer=args.use_hammer, use_llm=args.use_llm)
        result = try_test_data(server, agent, datum, max_steps=args.max_steps, max_trials_per_goal=args.max_trials_per_goal)
        if result is None:
            with open(placeholder_file_name, 'w') as f:
                json.dump({ 'id': datum['id'] }, f)
        else:
            if placeholder_file_name.is_file():
                placeholder_file_name.unlink()
            with open(file_name, 'w') as f:
                json.dump({ 'id': datum['id'], 'success': result.success, 'steps': result.steps  }, f)
