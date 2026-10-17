# Clustering and Optimisation for Classifier Ensembles

MATLAB experiments exploring clustering, data diversity and particle swarm optimisation for classifier ensembles.

## Research reference and attribution

- Jan, Z. M. and Verma, B. (2019). Ensemble Classifier Optimization by Reducing Input Features and Base Classifiers. IEEE Congress on Evolutionary Computation (CEC), pp. 1580–1587.
- Jan, Z. M., Verma, B. and Fletcher, S. (2018). Optimizing clustering to promote data diversity when generating an ensemble classifier. Genetic and Evolutionary Computation Conference Companion, pp. 1402–1409. ACM.

Please cite the relevant publication and retain source author notices. Existing dataset and third-party terms still apply; this cleanup does not grant a new licence.

## Data and environment

Extract `DTE.7z` into the repository root. Expected training files: `DTE/<name>/<name>-CV-tr-<fold>.mat`, with `dtrX` and `dtrY`. Test files use `-CV-ts-` and contain `dtsX` and `dtsY`. Fold numbers run from 1 to `params.numOfFolds`. Training and test labels must be column vectors.

**MATLAB dependencies:** Statistics and Machine Learning Toolbox; Deep Learning Toolbox; Global Optimization Toolbox.

No exact original MATLAB/toolbox version is recorded. Use MATLAB rather than assuming GNU Octave compatibility for the learner APIs.

## Start here

1. Open this repository as the MATLAB current folder. Use a separate session/path for each research repository: helper filenames overlap across projects.
2. Supply or extract the data described above. Check feature/label row counts and class labels.
3. Edit `Problem` in `mainProgram.m` (or `dataSetNames.m` where used), and start with one dataset and one repeat.
4. Run the synthetic utility checks, then inspect `mainProgram.m` before starting the full experiment:

```matlab
results = runtests('tests');
assertSuccess(results);
rng(42, 'twister');
mainProgram;
```

The seed is a starting point for recording runs; nested seed resets, parallel workers and toolbox versions can still affect reproducibility. `runMultiple.m` performs 50 full driver repetitions and is not the quick-start command.

## Workflow and output

`mainProgram.m` repeats experiments over `Problem`; `runTraining.m` loads pre-created folds and separates validation data. `generateClustersv2.m` constructs candidate clusters; `clusteringPSO.m` selects clusters; `trainClassifiers.m` and `classifierSelectionPSO.m` train and select models.

`saveResults.m` appends rows to `results.csv` in the current folder and writes the header only when creating the file. Driver-level CSV writes are sequential to avoid concurrent append races. Archive old output before a new configuration; rows do not contain a complete environment/configuration record. Committed CSV files are historical outputs, not freshly validated benchmarks.

## Review notes and reproducibility limits

Fold state is isolated and the restored SVM helper remains in place. ANN/RBF outputs retain the training label vocabulary, and the legacy pattern trainer now defines its input matrix. PSO uses its existing >0.6 threshold consistently during scoring and return; empty masks retain the highest-weight candidate. Prediction failures propagate instead of becoming class-one placeholder votes. The default driver repeats each dataset ten times, and `runMultiple.m` repeats that complete driver another 50 times.

## Correctness review — October 2026

This pass fixes implementation defects; it does not validate the associated research results. Changes can alter model pools, predictions and measured scores. Re-run the experiments before comparing corrected code with historical CSVs or published results.

- Label conversion uses numeric category names, not categorical indices.
- Voting rejects empty/nonfinite inputs; a prediction failure is reported rather than counted as a fabricated class-one vote.
- Neural models store their training labels in `net.userdata.classLabels`. Retrain old models or restore that exact training vocabulary before prediction; inferring it from a validation subset is unsafe.
- Synthetic checks cover normalization, CSV append behavior and the new prediction/selection boundaries. Additional cluster checks run where applicable.

See [the code review](docs/code-review.md) for repository-specific changes, tests and remaining limitations.

**Validation:** all MATLAB sources were statically parsed/linted with MISS_HIT. MATLAB and its learner toolboxes are unavailable in the review environment. The MATLAB regression suites and full training runs have **not been executed** here. Run `runtests('tests')` and `assertSuccess(results)` in MATLAB before using the corrected experiments.

[Profile and project directory](https://github.com/zohaibjan)
