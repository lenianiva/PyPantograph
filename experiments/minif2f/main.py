#!/usr/bin/env python3

import subprocess, json, argparse
from typing import Optional
from pathlib import Path
from pantograph.server import Server, ServerError
from pantograph.search import SearchResult
from model.llm_agent import LLMAgent

PATH_EXPERIMENT = Path(__file__).parent.resolve()

def get_project_and_lean_path():
    cwd = PATH_EXPERIMENT / 'MiniF2F'
    p = subprocess.check_output(['lake', 'env', 'printenv', 'LEAN_PATH'], cwd=cwd)
    return cwd, p

def read_test_data(use_valid: bool):
    jsonl_path = PATH_EXPERIMENT / ('valid.jsonl' if use_valid else 'test.jsonl')
    with open(jsonl_path, 'r') as f:
        return [json.loads(l) for l in list(f)]

def try_test_data(server, agent, entry: dict, max_steps: int, max_trials_per_goal: int) -> Optional[SearchResult]:
    command = entry["formal_statement"]
    print(command)
    informal_stmt = entry["informal_stmt"]
    informal_proof = entry["informal_proof"]

    goal_state, = server.load_sorry(command)
    try:
        return agent.search(
            server=server,
            goal_state=goal_state,
            informal_stmt=informal_stmt,
            informal_proof=informal_proof,
            verbose=True,
            max_steps=max_steps,
            max_trials_per_goal=max_trials_per_goal
        )
    except ServerError as e:
        return None

def output_file_name(datum, use_hammer: bool, use_llm: bool):
    name = datum["id"]
    folder = 'output'
    if use_hammer:
        folder += '-hammer'
    if use_llm:
        folder += '-llm'
    folder = PATH_EXPERIMENT / folder
    folder.mkdir(exist_ok=True, parents=True)
    return folder / f"{name}.json"

def dry_run(args):
    test_data = read_test_data(args.validation)
    for datum in test_data:
        print(datum["formal_statement"])

def run_eval(args):
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
        server = Server(imports=["MiniF2F"], project_path=project_path, lean_path=lean_path, options=["maxHeartbeats=0"])
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='MiniF2F Search',
                    description='Executes LLM on MiniF2F Search')
    parser.add_argument('--use-hammer', action='store_true')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="List the data used, but don't run")
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--use-llm', action='store_true')
    parser.add_argument('-s', '--max-steps', default=50)
    parser.add_argument('-t', '--max-trials-per-goal', default=2)
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args)
    else:
        run_eval(args)
