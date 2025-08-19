set positional-arguments

default:
  just --list

run *args:
  uv run python main.py {{ args }}

ingest *args:
  uv run python main.py ingest {{ args }}

search *args:
  uv run python main.py search {{ args }}

fmt:
  ruff format *.py losem/

sh:
  uv run ipython

clean:
  uv run pyclean . --verbose

clean-all:
  uv run pyclean . --verbose --debris

test:
  uv run pytest

check *file:
  uv run pyright {{ file }}

check-security:
  uv run bandit -c pyproject.toml -r *.py losem/
