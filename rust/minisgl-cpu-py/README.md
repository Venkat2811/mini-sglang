# minisgl-cpu-py

PyO3/maturin bindings for `minisgl-cpu-core`.

## Local build

```bash
cd rust/minisgl-cpu-py
maturin develop
```

## Smoke test

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
