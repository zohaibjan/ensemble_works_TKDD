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

## Still requires target-runtime validation

- MATLAB/toolbox compatibility and neural-network training/prediction APIs.
- Full training runs with the documented datasets, preserving training/validation/test separation.
- Numerical quality, random seeds, paper-method agreement, runtime and memory consumption.
- Optional historical optimizer/diversity/alternate-driver files are not a validated library.

The review environment has no MATLAB installation. Static lint passed, but the MATLAB unit tests were not run. No reproduced accuracy or benchmark claim is made.
