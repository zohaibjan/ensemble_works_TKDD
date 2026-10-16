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

The default driver repeats each dataset ten times; `runMultiple.m` repeats the entire driver another 50 times. Start with one dataset and one repeat. Fold state is now isolated: models from previous folds are discarded. The missing SVM helper was restored from the companion IEEE_TKDE implementation, and the ANN call now matches the local two-argument trainer. These corrections require new benchmark runs before comparing with stored or published results. ANN output indices require a consistent 1-based label mapping.

The cleanup aligns function/file names, restores visible warnings, makes CSV handles close on errors, and tests constant-feature normalisation plus CSV append behaviour. MATLAB experiments and the new utility tests have **not** been executed in the cleanup environment, which has no MATLAB installation. Static parsing is not a reproduction of the paper.

[Profile and project directory](https://github.com/zohaibjan)
