# MiniF2F

This is an experiment on running a LLM prover on miniF2F data. Build the project
`MiniF2F` with `lake build`. Check the environment and data with

``` sh
python3 experiments/minif2f/main.py check
python3 experiments/minif2f/main.py list
```

and run experiments with

```sh
python3 experiments/minif2f/main.py eval [--use-llm] [--use-hammer]
```

Read the help message carefully.

## Developing

Run unit tests with

``` sh
python3 -m model.{llm_agent,gen_tactic}
```

