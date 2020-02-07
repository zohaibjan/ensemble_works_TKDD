function results = runTraining(p_name , params)
warning('off','all');

nonOptimized_Accuracy = [];
optimized_Accuracy = [];
classifierIndex = 1;
classifiers = {};
data = [];


for f=1:10
    data=load([pwd,filesep,'DTE',filesep,p_name,filesep,p_name,'-CV-tr-', num2str(f)]);
    X=data.dtrX; Y=data.dtrY;
    traindata = [X Y];
   
    testdata=load([pwd,filesep,'DTE',filesep,p_name,filesep,p_name,'-CV-ts-', num2str(f)]);
    testX = testdata.dtsX;
    testy = testdata.dtsY;
       
    %% SEPARATE VALIDATION DATA
    cvv = cvpartition(trainData(:,end), 'holdout', 0.1);
    idxs = cvv.test;
    valData = trainData(idxs,:);
    trainData = trainData(~idxs, :);
    
    trainX = trainData(:, 1:end-1);
    trainy = trainData(:, end);
    
    valX = valData(:, 1:end-1);
    valy = valData(:, end);
    
    allClusters = generateClustersv2([trainX, trainy], params);
    
    bestClusters = clusteringPSO(allClusters, [valX  valy], params);
    bestClusters = find(bestClusters.chromosome);
    selectedClusters = {};
    
    for i=1:length(bestClusters)
        selectedClusters{1,i} = allClusters{1, bestClusters(i)};
    end
    
    for c=selectedClusters
        X = c{1,1}(:, 1:end-1);
        y = c{1,1}(:, end);
        %         classifiers{classifierIndex} =b getCNN(X, y);
        all = trainClassifiers(X, y, params);
        for temp = 1:length(all)
            classifiers{classifierIndex} = all{1,temp};
            classifierIndex = classifierIndex + 1;
        end
    end
    
    psoEnsemble = classifierSelectionPSO(classifiers, [valX, valy]);
    psoEnsemble = find(psoEnsemble.chromosome);
    selectedClassifiers = {};
    
    for i=1:length(psoEnsemble)
        selectedClassifiers{1,i} = classifiers{1, psoEnsemble(i)};
    end
    
    nonOptimized_Accuracy(f) = fusion(classifiers, [testX, testy]);
    optimized_Accuracy(f) = fusion(selectedClassifiers, [testX, testy]);
end
results.nonOptimized_Accuracy = mean(nonOptimized_Accuracy);
results.nonOptimized_stdDEV = std(nonOptimized_Accuracy);
results.optimized_Accuracy = mean(optimized_Accuracy);
results.optimized_stdDEV = std(optimized_Accuracy);
end

