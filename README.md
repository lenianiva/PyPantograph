# PyPantograph

A Machine-to-Machine Interaction System for Lean 4.

## Installation

1. Install `poetry`
2. Clone this repository with submodules:
```sh
git clone --recurse-submodules <repo-path>
```
3. Install `elan` and `lake`: See [Lean Manual](https://docs.lean-lang.org/lean4/doc/setup.html)
4. Execute
```sh
poetry build
poetry install
```

## Documentation

Build the documentations by
```sh
jupyter-book build docs
```
Then serve
```sh
cd docs/_build/html
python3 -m http.server -d .
```

## Examples

For API interaction examples, see `examples/README.md`. The examples directory
also contains a comprehensive Jupyter notebook.

## Experiments

In `experiments/`, there are some experiments:
1. `minif2f` is an example of executing a `sglang` based prover on the miniF2F dataset
2. `dsp` is an Lean implementation of Draft-Sketch-Prove

The experiments should be run in `poetry shell`. The environment variable
`OPENAI_API_KEY` must be set when running experiments calling the OpenAI API.

## Referencing

[Paper Link](https://arxiv.org/abs/2410.16429)

```bib
@misc{pantograph,
      title={Pantograph: A Machine-to-Machine Interaction Interface for Advanced Theorem Proving, High Level Reasoning, and Data Extraction in Lean 4},
      author={Leni Aniva and Chuyue Sun and Brando Miranda and Clark Barrett and Sanmi Koyejo},
      year={2024},
      eprint={2410.16429},
      archivePrefix={arXiv},
      primaryClass={cs.LO},
      url={https://arxiv.org/abs/2410.16429},
}
```
