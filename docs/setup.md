# Setup

Install `poetry`. Then, run
```sh
poetry build
```
This builds a wheel of Pantograph which can then be installed.

To run the examples and experiments, setup a poetry shell:
```sh
poetry install
poetry shell
```
This drops the current shell into an environment where the development packages are available.

All interactions with Lean pass through the `Server` class. Create an instance
with
```python
from pantograph import Server
server = Server()
```

## Lean Dependencies

To use external Lean dependencies such as
[Mathlib4](https://github.com/leanprover-community/mathlib4), Pantograph relies
on an existing Lean repository. Instructions for creating this repository can be
found [here](https://docs.lean-lang.org/lean4/doc/setup.html#lake).
