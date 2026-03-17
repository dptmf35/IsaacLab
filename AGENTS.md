# Repository Guidelines

## Project Structure & Module Organization
Core code lives under `source/`, split into extensions: `source/isaaclab` (framework), `source/isaaclab_tasks` (task environments), `source/isaaclab_rl` (RL wrappers), `source/isaaclab_mimic`, and `source/isaaclab_contrib`.  
Tests are colocated per extension in `source/*/test/` with filenames like `test_*.py`.  
Executable workflows are in `scripts/` (for example, `scripts/demos/` and `scripts/reinforcement_learning/`).  
Project tooling is in `tools/` (test runners, settings), app launch configs in `apps/`, and docs sources in `docs/`.

## Build, Test, and Development Commands
Use the repository launcher so Isaac Sim Python/environment resolution stays consistent:

- `./isaaclab.sh -i` installs project extensions and RL extras into the active env.
- `./isaaclab.sh -p scripts/demos/arms.py` runs a Python script with Isaac Lab’s Python.
- `./isaaclab.sh -f` runs `pre-commit` hooks (ruff lint + format, codespell, YAML/TOML checks).
- `./isaaclab.sh -t` runs pytest via the project test harness in `tools/`.
- `./isaaclab.sh -p tools/run_all_tests.py --discover_only` lists all discovered tests.
- `./isaaclab.sh -d` builds Sphinx docs from `docs/`.

## Coding Style & Naming Conventions
Python 3.11 is the target runtime. Formatting/linting is configured in `pyproject.toml`:

- Ruff line length: 120; isort ordering is enforced through Ruff.
- Prefer `snake_case` for modules/functions/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Keep imports grouped according to configured extension sections (`isaaclab`, `isaaclab_tasks`, etc.).
- Run `./isaaclab.sh -f` before pushing.

## Testing Guidelines
Pytest is the test framework (`[tool.pytest.ini_options]` in `pyproject.toml`).  
Add tests in the corresponding extension’s `test/` directory and follow `test_*.py` naming.  
Use targeted runs during development (for example, `./isaaclab.sh -p -m pytest source/isaaclab/test/controllers`).  
For full validation, use `tools/run_all_tests.py` and extension filtering/timeout flags when needed.

## Commit & Pull Request Guidelines
Recent commits follow short, imperative summaries with optional PR references, e.g. `Fixes ... (#4507)`.  
Keep commit scope focused and include rationale for behavior changes.  
PRs should include: clear description, linked issue/PR context, test evidence (command + result), and screenshots/video for UI or simulation behavior changes.  
Sign contributions in accordance with the repository’s DCO expectations in `CONTRIBUTING.md`.
