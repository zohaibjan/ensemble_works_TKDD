# Executed validation

Date: 17 October 2026. [Successful GitHub Actions run 39000461858](https://github.com/zohaibjan/ensemble_works_TKDD/actions/runs/39000461858) at commit `5ca4ac3748b9718c11e9878f67dc8b491da48f03`.

- MATLAB R2024b on Ubuntu 22.04.
- 18 tests passed; zero failed or incomplete.
- 29 root function files loaded/compiled by MATLAB without executing the experiment entry scripts.
- Installed toolboxes: Statistics and Machine Learning, Deep Learning, Global Optimization, Optimization (dependency), and Parallel Computing.

## What was executed

Actual SVM/tree fitting, ANN label preservation, particleswarm score/selection consistency, voting, selection, normalization and CSV utilities.

Toolbox tests fit real models and/or call real solvers. Deterministic predict-compatible fixtures are also used to isolate selection and voting behavior. Smoke tests use small synthetic datasets and bounded solver/epoch budgets; default research budgets remain available.

## Changes and behavior

PSO now accepts optional solver options for bounded smoke tests while retaining the historical >0.6 selection threshold.

## Reproduce

Use a clean MATLAB session with only this repository on the path:

```matlab
results = runtests('tests');
assertSuccess(results);
```

The workflow runs tests on MATLAB changes and pull requests, and supports manual dispatch. It has read-only repository permissions and performs no deployment.

## Limits

These tests do not execute every function branch or reproduce a paper's full datasets, published accuracy, runtime or memory results. Experiment scripts and unsupported auxiliary paths still need their documented inputs/dependencies. Root function loading is a compilation check, not full code coverage. Consult [code-review.md](code-review.md) for repository-specific limitations. Historical result CSVs were not regenerated or relabelled.
