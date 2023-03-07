# mlflow-template

Personal MLflow + Hydra template

# Prerequisites

- > = Python 3.10

# Package Management

## Install packages

```
python3 -m pip install -e .
```

## Install dev packages

```
python3 -m pip install -e ".[dev]"
```

## Additional tools for package management

```
python3 -m pip install pip-tools pip-autoremove
```

## Generate requirements.txt files using pip-tools

```
pip-compile -o requirements.txt pyproject.toml
pip-compile --extra dev -o requirements-dev.txt pyproject.toml
```
