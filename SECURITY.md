# Security Policy

## Supported versions

jNO follows a single-track release model — only the latest tagged
release on PyPI receives security fixes. The `main` branch is the
development line; security patches land there and ship in the next
minor release.

| Version            | Supported          |
|--------------------|--------------------|
| Latest `main`      | :white_check_mark: |
| Latest PyPI release| :white_check_mark: |
| All older releases | :x:                |

## Reporting a vulnerability

If you believe you have found a security vulnerability in jNO,
**please do not file a public GitHub issue**. Public disclosure before
a fix is available puts other users at risk.

Instead, report privately by one of:

- Email the maintainer at
  [leon.armbruster@iisb.fraunhofer.de](mailto:leon.armbruster@iisb.fraunhofer.de)
  with `[jno security]` in the subject line.
- Use GitHub's
  [private vulnerability reporting](https://github.com/FhG-IISB/jno/security/advisories/new)
  (Security tab → "Report a vulnerability").

Please include:

1. A description of the vulnerability and what it allows.
2. The shortest reproduction you have (a single Python file, a domain
   construction, a malicious input file, etc.).
3. The jNO version (`python -c "import jno; print(jno.__version__)"`)
   and any relevant JAX / hardware details.
4. Whether you would like to be credited and how.

We aim to acknowledge reports within 5 business days, share a
remediation plan within 14 days, and ship a fix in the next minor
release. For severe issues we may publish a patch release sooner.

## Scope

The most security-relevant surface in jNO is:

- Loading checkpoints and pickled artefacts (`jno.save`, `jno.load`,
  `statistics.load`) — these use `cloudpickle` and execute arbitrary
  Python on load. Treat any checkpoint from an untrusted source as
  potentially malicious code.
- Loading meshes via `meshio` from user-supplied paths.
- The hyperparameter tuner running arbitrary user-supplied training
  configs.

Out of scope:

- Bugs that require already running arbitrary code in the same Python
  process as jNO (e.g., a malicious script importing `jno` and then
  calling its own functions). Those are application-level issues.
- Denial of service through excessive compute / memory in
  user-supplied PDEs (this is the user's responsibility — JAX gives
  them the gun, jNO just provides the holster).
