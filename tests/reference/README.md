# Reference data from independent RCWA libraries

`rcwa_reference.json` holds T and R from **torcwa** and **fmmax** for the cases in `cases.py`, each solved with
Laurent's rule and the parallelogramic truncation `|m|, |n| <= M`. `tests/test_rcwa_reference.py` checks that
jNO, given the same problem (`formulation="FFT"`, `Truncation.PARALLELOGRAMIC`), reproduces them. The test needs
no reference library; only regenerating the data does.

The libraries never enter jNO's own environment. They live in a separate pixi project, whose manifest is copied
here as `rcwa-ref.pixi.toml` (Python 3.12, fmmax 1.7.1, Meent, torcwa, CPU-only torch):

    mkdir -p ~/jNO/rcwa-ref && cp tests/reference/rcwa-ref.pixi.toml ~/jNO/rcwa-ref/pixi.toml
    cd ~/jNO/rcwa-ref && pixi install
    pixi run --manifest-path ~/jNO/rcwa-ref/pixi.toml python tests/reference/generate_reference.py

Notes:

- torcwa samples the permittivity at pixel corners, jNO and fmmax at pixel centres. That shifts the structure
  by half a pixel, which leaves every diffraction efficiency unchanged.
- Cases use a pure s or p input, so no library's s/p sign convention enters the comparison.
- fmmax is compared at normal incidence only; at oblique incidence it does not conserve energy here (git-bug
  a20b207).
- Not yet included: Meent (its 2-D factorization rule still has to be matched) and S4 (needs a source build).
