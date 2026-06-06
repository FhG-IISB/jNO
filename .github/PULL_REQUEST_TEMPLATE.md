<!--
Thanks for contributing to jNO!

Please fill in the sections below. Anything irrelevant can be deleted.
-->

## Summary

<!-- One or two sentences: what does this PR change, and why? -->

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that changes existing behaviour)
- [ ] Documentation only
- [ ] Test or tooling only

## Linked issues / PRs

<!-- Closes #123, Refs #456 -->

## Pre-merge checklist

- [ ] `pixi run fmt` passes (no formatting changes)
- [ ] `pixi run lint` passes (no warnings)
- [ ] `pixi run test` passes (or specify which tests are intentionally skipped/slow)
- [ ] Public-API change → docs updated (`docs/*.md`, docstrings, [Concepts page](../docs/Glossary.md))
- [ ] If touching `jno.core`, `jno.domain`, `jno.trace`, or `jno.trace_compiler`: added or extended a test under `tests/test_*.py`
- [ ] If user-visible behaviour changed: changelog entry / migration note in PR body
- [ ] No accidental `print(...)` / `breakpoint()` / temporary files committed

## Testing notes

<!--
What did you actually run? Examples:
- pixi run pytest tests/test_core.py -x -q
- pixi run python docs/tutorial_examples/01_basics/poisson_1d.py
- Manual smoke on GPU: …
-->

## Reviewer focus

<!--
Anything specific you want reviewers to look at? Performance? API
design? A particular edge case? Leave blank for general review.
-->
