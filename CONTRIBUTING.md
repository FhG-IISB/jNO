# Contributing to jNO

Thank you for your interest in contributing to jNO! This document outlines the development workflow and quality standards.

## Environment Setup

jNO uses [pixi](https://pixi.sh) to manage the development environment. Install it once:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then clone the repo and install all dependencies from the lock file:

```bash
git clone https://github.com/FhG-IISB/jNO.git
cd jNO
pixi install
```

That's it — no manual conda/pip steps needed.

## Pre-commit hooks (recommended)

Install the git pre-commit hooks once per checkout. They run `ruff
format`, `ruff check --fix`, and a handful of housekeeping checks
(trailing whitespace, large files, YAML/TOML validity) on every commit:

```bash
pixi run pre-commit install
```

After installation, every `git commit` will auto-format and lint the
files you're about to commit. If a hook modifies a file, the commit is
aborted — re-stage and commit again.

You can run all hooks against the entire repo manually:

```bash
pixi run pre-commit run --all-files
```

The hooks pin the same ruff version as the CI pixi environment, so
local and CI behaviour stay aligned.

## Pre-PR Checklist

Before submitting code or pushing changes, run the following checks locally using the pixi tasks defined in `pyproject.toml`.

### 1. Code Formatting

```bash
pixi run fmt
```

Auto-formats all Python files with [ruff](https://docs.astral.sh/ruff/). Re-run until no files are reformatted.

### 2. Linting

```bash
pixi run lint
```

Runs `ruff check --fix` over the codebase. Fix any remaining errors that cannot be auto-fixed before committing.

### 3. Unit Tests

```bash
pixi run test
```

Runs the fast test suite (`pytest -x --tb=short`). All tests must pass before submitting a pull request.

## Development Workflow

1. **Create a feature branch**: start from `main` and use a descriptive name.
2. **Implement changes**: follow the project conventions in the existing code.
3. **Run checks locally**: execute `fmt`, `lint`, and `test` in order.
4. **Commit and push**: once all checks pass, commit with a clear message.
5. **Submit PR**: request review and address any feedback.

## Quick Check — All Three in Sequence

```bash
pixi run fmt && pixi run lint && pixi run test
```

## Available Pixi Tasks

| Task | Command | Purpose |
|------|---------|---------|
| `pixi run fmt` | `ruff format .` | Auto-format code |
| `pixi run lint` | `ruff check . --fix` | Lint and auto-fix |
| `pixi run test` | `pytest -x --tb=short` | Run fast test suite |
| `pixi run ci-fmt` | `ruff format --check .` | Format check (read-only, used by CI) |
| `pixi run ci-lint` | `ruff check .` | Lint check (no auto-fix, used by CI) |
| `pixi run ci-test` | `pytest -x --tb=short -m 'not slow'` | Test suite without slow tests (used by CI) |

## Writing docs

Docs live in `docs/` and build with `mkdocs build --strict`, which **must pass with zero warnings**
before a PR. `--strict` catches dead internal links, unresolved cross-references, and malformed
docstrings (a documented parameter with no annotation on the signature will fail it).

Two conventions keep the pages readable as they grow:

- **Callouts are typed.** Seven types, each with one job — `abstract`, `note`, `tip`, `measured`,
  `fun-fact`, `warning`, `danger`. A reader should be able to tell from the colour alone whether a
  block is skippable background, advice to take, or a trap. The full convention, with each type
  rendered, is at [Docs style — callouts](docs/misc/callouts.md).
- **Collapse depth, don't cut it.** Swap `!!!` for `???` and the block folds. Derivations, benchmark
  tables and rationale go there — write as much as you want, and the page stays scannable. A page
  should read completely with every collapsed block shut. The one exception: a `danger` block never
  collapses.

Keep paragraphs under ~90 words. Anything enumerable — element families, what a slot supports, which
forms are parametric — is a table, not prose.

## CI/CD

Pull requests are checked by `.github/workflows/ci.yml`, which runs `ci-fmt`, `ci-lint`, and `ci-test` via pixi. The CI environment is identical to local — same lock file, same tool versions, no surprises.
