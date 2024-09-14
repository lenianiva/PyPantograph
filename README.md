# PyPantograph

Python interface to the Pantograph library

## Getting started
Update submodule
``` bash
git submodule update --init
```
Install dependencies
```bash
poetry install
```

Execute
```bash
poetry build
```
To run server tests:
``` bash
python -m pantograph.server
python -m pantograph.search
```
The tests in `pantograph/server.py` also serve as simple interaction examples

## Examples

For API interaction examples, see `examples/README.md`

An agent based on the `sglang` library is provided in
`pantograph/search_llm.py`. To use this agent, set the environment variable
`OPENAI_API_KEY`, and run
```bash
python3 -m pantograph.search_llm
```

## Experiments

In `experiments/`, there is an experiment on running a LLM prover on miniF2F
data. Run with

```sh
python3 experiments/miniF2F_search.py [--dry-run]
```

## Referencing

```bib
@misc{pantograph,
	title = "Pantograph, A Machine-to-Machine Interface for Lean 4",
	author = {Aniva, Leni and Miranda, Brando and Sun, Chuyue},
	year = 2024,
	howpublished = {\url{https://github.com/lenianiva/PyPantograph}}
}
```
