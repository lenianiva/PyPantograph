# Setup

Install `poetry`. Then, run
```sh
poetry build
```

This builds a wheel of Pantograph in `dist` which can then be installed. For
example, a downstream project could have this line in its `pyproject.toml`

```toml
pantograph = { file = "path/to/wheel/dist/pantograph-0.2.19-cp312-cp312-manylinux_2_40_x86_64.whl" }
```

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

The server created from `Server()` is sufficient for basic theorem proving tasks
reliant on Lean's `Init` library. Some users may find this insufficient and want
to use non-builtin libraries such as Aesop or Mathlib4.

To use external Lean dependencies such as
[Mathlib4](https://github.com/leanprover-community/mathlib4), Pantograph relies
on an existing Lean repository. Instructions for creating this repository can be
found [here](https://docs.lean-lang.org/lean4/doc/setup.html#lake).

After creating this initial Lean repository, execute in the repository
```sh
lake build
```

to build all files from the repository. This step is necessary after any file in
the repository is modified.

Then, feed the repository's path to the server
```python
server = Server(project_path="./path-to-lean-repo/")
```

For a complete example, see `examples/`.
