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

See `examples/README.md`

## Referencing

```bib
@misc{pantograph,
	title = "Pantograph, A Machine-to-Machine Interface for Lean 4",
	author = {Aniva, Leni and Miranda, Brando and Sun, Chuyue},
	year = 2024,
	howpublished = {\url{https://github.com/lenianiva/PyPantograph}}
}
```
