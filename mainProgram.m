function program = mainProgram()

addpath(genpath('P-Data'));
Problem={'breast-cancer-wisconsin','diabetic_retinopathy','ecoli',...
           'haberman','ionosphere','iris','liver',...
           'pima_diabetec','segment2',...
           'sonar','thyroid','vehicle','wine'};

% Problem = {'wine'};
%
%% Model SETTINGS
params.numOfRuns = 10;
params.numOfFolds = 10;                   % Create CROSS VALIDATION FOLDS
params.classifiers = {'ANN', 'KNN', 'DT', 'DISCR','NB','SVM'};
params.trainFunctionANN={'trainlm','trainbfg','trainrp','trainscg','traincgb','traincgf','traincgp','trainoss','traingdx'};
params.trainFunctionDiscriminant = {'pseudoLinear','pseudoQuadratic'};
params.kernelFunctionSVM={'gaussian','polynomial','linear'};

%% MAIN LOOP
parfor i = 1:length(Problem)
    results = {};
    nonOptimized_Accuracy = [];
    optimized_Accuracy = [];
	p_name = [];
    for runs = 1:params.numOfRuns
        p_name = Problem{i};
        results = runTraining(p_name, params);
        nonOptimized_Accuracy(runs) = results.nonOptimized_Accuracy;
		optimized_Accuracy(runs) = results.optimized_Accuracy
    end
    results.p_name = p_name;
    results.nonOptimized_Accuracy = mean(nonOptimized_Accuracy);
    results.optimized_Accuracy = mean(optimized_Accuracy);
    results.nonOptimized_stdDEV = std(nonOptimized_Accuracy);
    results.optimized_stdDEV = std(optimized_Accuracy);
    saveResults(results);
end
end

