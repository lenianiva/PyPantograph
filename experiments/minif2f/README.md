# MiniF2F

This is an experiment on running a LLM prover on miniF2F data. Build the project
`MiniF2F` with `lake build`, and run with

```sh
python3 experiments/minif2f/main.py [--dry-run] [--use-llm]
```

Read the help message carefully.

## Developing

Run unit tests with

``` sh
python3 -m model.{llm_agent,gen_tactic}
```

