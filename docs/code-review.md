# Code review and validation scope

Fold state is isolated and the restored SVM helper remains in place. ANN/RBF outputs retain the training label vocabulary, and the legacy pattern trainer now defines its input matrix. PSO uses its existing >0.6 threshold consistently during scoring and return; empty masks retain the highest-weight candidate. Prediction failures propagate instead of becoming class-one placeholder votes. The default driver repeats each dataset ten times, and `runMultiple.m` repeats that complete driver another 50 times.

## Changed behavior and acceptance

The fixes preserve original source attribution and the documented experiment entry points. They change previously erroneous edge cases and may change reported results. Stored outputs were not rewritten or relabelled as new results.

Run in a clean MATLAB session with only this repository on the path:

```matlab
results = runtests('tests');
assertSuccess(results);
```

The test fixtures are synthetic. `FixedClassifier`, where supplied, implements `predict` without fitting a model and is on the path only for the test session. Majority ties choose the smallest numeric label. Empty or invalid votes fail explicitly. ANN vocabularies are learned from training labels, not held-out labels. PSO thresholds retain the repository's setting, with one shared nonempty mask policy for objective scoring and final return. Random selection, where used, has at most pool-size times elimination-budget attempts.

## Executed MATLAB validation

Actual SVM/tree fitting, ANN label preservation, particleswarm score/selection consistency, voting, selection, normalization and CSV utilities.

PSO now accepts optional solver options for bounded smoke tests while retaining the historical >0.6 selection threshold.

See [validation.md](validation.md) for the exact run, release and boundaries.

## Remaining validation

- Compatibility beyond the tested MATLAB R2024b/toolbox release.
- Full training runs with the documented datasets, preserving training/validation/test separation.
- Numerical quality, random seeds, paper-method agreement, runtime and memory consumption.
- Optional historical optimizer/diversity/alternate-driver files are not a validated library.

The regression/smoke suite passed in MATLAB R2024b using MathWorks GitHub Actions. Synthetic fixtures check implementation behavior; full research experiments, historical accuracy and performance benchmarks have not been reproduced.
