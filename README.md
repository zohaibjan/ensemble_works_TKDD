# Clustering and Optimisation for Classifier Ensembles

MATLAB research experiments exploring data diversity, clustering, feature reduction and base-classifier selection.

## Associated publications

- Jan, Z. M. and Verma, B. (2019). Ensemble Classifier Optimization by Reducing Input Features and Base Classifiers. IEEE Congress on Evolutionary Computation (CEC), pp. 1580–1587.
- Jan, Z. M., Verma, B. and Fletcher, S. (2018). Optimizing clustering to promote data diversity when generating an ensemble classifier. Genetic and Evolutionary Computation Conference Companion, pp. 1402–1409. ACM.

Please cite the relevant publication when using this work.

## Getting started

1. Inspect and extract `DTE.7z` if the expected `DTE` directory is not present.
2. Open `mainProgram.m` in MATLAB and review dataset and experiment settings.
3. Ensure the optimisation and neural-network toolboxes required by the source are installed.
4. Run `mainProgram`.

The original dataset notes identify UCI datasets converted to MAT files, with pre-created ten-fold splits. Check the original dataset terms and the actual split files before reuse.

## Code map

Clustering helpers feed `runTraining.m`; `trainClassifiers.m` builds base learners; the fusion and PSO helpers select/combine predictions; `saveResults.m` exports results.

## Status

Historical research implementation. Repository documentation has been organised; experiments have not been rerun or independently reproduced.

[Profile and project directory](https://github.com/zohaibjan)
