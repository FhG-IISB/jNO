# Contributing to jNO

Thank you for your interest in contributing to jNO! This document outlines the development workflow and quality standards.

## Pre-Commit Checklist

Before submitting code or pushing changes, please run the following checks locally:

### 1. Code Formatting

Format all Python code with [Black](https://github.com/psf/black):

```bash
black jno/
```

### 2. Type Checking

Run mypy to ensure type safety:

```bash
uv run mypy jno/ --ignore-missing-imports --no-strict-optional
```

All errors must be resolved before committing. Notes and warnings are acceptable.

### 3. Unit Tests

Run the full test suite to ensure no regressions:

```bash
uv run pytest
```

All tests must pass before submitting a pull request.

## Development Workflow

1. **Create feature branch**: Start from `main` and create a descriptive branch.
2. **Implement changes**: Write code following project conventions.
3. **Run checks locally**: Execute the three commands above in order.
4. **Commit and push**: Once all checks pass, commit with a clear message.
5. **Submit PR**: Request review and address any feedback.

## Quick Check Command

To run all three checks in sequence:

```bash
black jno/ && uv run pytest && uv run mypy jno/ --ignore-missing-imports --no-strict-optional
```

## CI/CD

These checks are also enforced via continuous integration. All commits must pass automated checks before merging.
