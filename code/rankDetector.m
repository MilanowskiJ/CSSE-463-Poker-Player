

function [rank] = rankDetector()

%% Reset and Read in Images
% clear all; close all; clc;

% Set Director Paths
rootDir = 'C:\Users\boewebe\Documents\CSSE463-ImageRecognition\Project\8-13\CSSE-463-Poker-Player-main';
trainDir = [rootDir '\dataset\custom_train\rank'];
testDir = [rootDir '\dataset\custom_test\rank'];

% rootDir = 'C:\Users\boewebe\Documents\CSSE463-ImageRecognition\Project\8-16\CSSE-463-Poker-Player-main';
% trainDir = [rootDir '\dataset\croppedInputData\suit'];
% testDir = [rootDir '\dataset\custom_test\suit'];

% For Hough Transform Function - Create new folder to store rotated
% images, and clear folder of files if folder exists already
newTestImgDir = [testDir '\new'];
mkdir(newTestImgDir);
delete([newTestImgDir '\*']);

newTrainImgDir = [trainDir '\new'];
mkdir(newTrainImgDir);
delete([newTrainImgDir '\*']);

% Load originial images
trainImages = imageDatastore(...
   trainDir, ...
   'IncludeSubfolders',true, ...
   'LabelSource', 'foldernames');
origTrainLabels = trainImages.Labels(:);

% [trainData, testData] = splitEachLabel(trainImages,0.8,'randomize');
% trainImages = trainData;
% testImages = testData;

testImages = imageDatastore(...
   testDir, ...
   'IncludeSubfolders',true, ...
   'LabelSource', 'foldernames');
origTestLabels = testImages.Labels(:);

% Rotate Images using correctImageRotation file
for i = 1:length(testImages.Files(:))
    img = imread(char(testImages.Files(i)));
    newImg = correctImageRotation(img);
    newImg = uint8(newImg);
    
    if isa(newImg, 'uint16')
    	newImg = uint8(newImg/256);
    end
    
    saveTo = [newTestImgDir '\' int2str(i) '.jpg'];
    imwrite(newImg, saveTo);
end

for i = 1:length(trainImages.Files(:))
    img = imread(char(trainImages.Files(i)));
    newImg = correctImageRotation(img);
    
    if isa(newImg, 'uint16')
    	newImg = uint8(newImg/256);
    end
    
    saveTo = [newTrainImgDir '\' int2str(i) '.jpg'];
    imwrite(newImg, saveTo);
end

% Load rotated images
testImages = imageDatastore(...
   newTestImgDir, ...
   'IncludeSubfolders',true, ...
   'LabelSource', 'foldernames');
testImages.Labels = origTestLabels;

trainImages = imageDatastore(...
   newTrainImgDir, ...
   'IncludeSubfolders',true, ...
   'LabelSource', 'foldernames');
trainImages.Labels = origTrainLabels;



% Load Network - AlexNet
net = alexnet;
inputSize = net.Layers(1).InputSize;
augmentedTestImages = augmentedImageDatastore(inputSize(1:2),testImages,'ColorPreprocessing', 'gray2rgb');%, 'DataAugmentation',imageAugmenter);

% Initialize variable to hold scores
finalScores = zeros(length(testImages.Labels), 13);
allLabels = ["2"; "3"; "4"; "5"; "6"; "7"; "8"; "9"; "10"; "A"; "J"; "Q"; "K"];


%% 2's:
%% CNN Feature Extraction :
label = '2';
% Get filepaths and labels for images chosen for the training dataset
[files, labels] = getTrainingDataset(trainImages, label, allLabels);
imgTrain = imageDatastore(...
    files, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');
imgTrain.Labels = labels;

% Update training image labels to be x vs. non-x
imgTrain.Labels = updateImgLabels(imgTrain, label);
imgTestLabels = updateImgLabels(testImages, label);

% Get features from the training data
[trainingFeatures, net] = getSuitFeatures(imgTrain);

%% Optimize Hyperparameters

% Change # to size of training data passed in
% c = cvpartition(length(imgTrain.Labels),'KFold',10);
% opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
%     'AcquisitionFunctionName','expected-improvement-plus');
% SVMModel = fitcsvm(trainingFeatures,imgTrain.Labels,'Standardize', true, 'KernelFunction','rbf',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)

% % Best Observed - BC: 272.21, KS: 96.184
% % Best Estimate - BC: 338.81, KS: 95.091

% Training
classifier = fitcsvm(trainingFeatures,imgTrain.Labels,'KernelFunction','linear','Standardize',true, ...
  'ClassNames',{'N2' '2'}, 'BoxConstraint', 338.81, 'KernelScale', 95.091);


%% Classify Test Images
featureLayer = 'fc7';
testFeatures = activations(net, augmentedTestImages, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
[predictedLabels, scores] = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

finalScores(:,1) = scores(:,2);

%% Determine Accuracy
twosAccuracy = getTestAccuracy(predictedLabels, imgTestLabels)





% Same process is used for all other ranks below







%% 3'S:
%% CNN Feature Extraction :
label = '3';
[files, labels] = getTrainingDataset(trainImages, label, allLabels);
imgTrain = imageDatastore(...
    files, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');
imgTrain.Labels = labels;

imgTrain.Labels = updateImgLabels(imgTrain, label);
imgTestLabels = updateImgLabels(testImages, label);

[trainingFeatures, net] = getSuitFeatures(imgTrain);

%% Optimize Hyperparameters

% Change # to size of training data passed in
% c = cvpartition(length(imgTrain.Labels),'KFold',10);
% opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
%     'AcquisitionFunctionName','expected-improvement-plus');
% SVMModel = fitcsvm(trainingFeatures,imgTrain.Labels,'Standardize', true, 'KernelFunction','rbf',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)

% % Best Observed - BC: 35.268, KS: 66.34
% % Best Estimate - BC: 243.69, KS: 61.737

% Training
classifier = fitcsvm(trainingFeatures,imgTrain.Labels,'KernelFunction','linear','Standardize',true, ...
  'ClassNames',{'N3' '3'}, 'BoxConstraint', 243.69, 'KernelScale', 61.737);


%% Classify Test Images
featureLayer = 'fc7';
testFeatures = activations(net, augmentedTestImages, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
[predictedLabels, scores] = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

finalScores(:,2) = scores(:,2);

%% Determine Accuracy
threesAccuracy = getTestAccuracy(predictedLabels, imgTestLabels)


















%% 4's:
%% CNN Feature Extraction :
label = '4';
[files, labels] = getTrainingDataset(trainImages, label, allLabels);
imgTrain = imageDatastore(...
    files, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');
imgTrain.Labels = labels;

imgTrain.Labels = updateImgLabels(imgTrain, label);
imgTestLabels = updateImgLabels(testImages, label);

[trainingFeatures, net] = getSuitFeatures(imgTrain);

%% Optimize Hyperparameters

% Change # to size of training data passed in
% c = cvpartition(length(imgTrain.Labels),'KFold',10);
% opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
%     'AcquisitionFunctionName','expected-improvement-plus');
% SVMModel = fitcsvm(trainingFeatures,imgTrain.Labels,'Standardize', true, 'KernelFunction','rbf',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)

% % Best Observed - BC: 91.618, KS: 78.461
% % Best Estimate - BC: 91.618, KS: 78.461

% Training
classifier = fitcsvm(trainingFeatures,imgTrain.Labels,'KernelFunction','linear','Standardize',true, ...
  'ClassNames',{'N4' '4'},'BoxConstraint', 91.618, 'KernelScale', 78.461);


%% Classify Test Images
featureLayer = 'fc7';
testFeatures = activations(net, augmentedTestImages, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
[predictedLabels, scores] = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

finalScores(:,3) = scores(:,2);

%% Determine Accuracy
foursAccuracy = getTestAccuracy(predictedLabels, imgTestLabels)













%% 5's:
%% CNN Feature Extraction :
label = '5';
[files, labels] = getTrainingDataset(trainImages, label, allLabels);
imgTrain = imageDatastore(...
    files, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');
imgTrain.Labels = labels;

imgTrain.Labels = updateImgLabels(imgTrain, label);
imgTestLabels = updateImgLabels(testImages, label);

[trainingFeatures, net] = getSuitFeatures(imgTrain);

%% Optimize Hyperparameters

% Change # to size of training data passed in
% c = cvpartition(length(imgTrain.Labels),'KFold',10);
% opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
%     'AcquisitionFunctionName','expected-improvement-plus');
% SVMModel = fitcsvm(trainingFeatures,imgTrain.Labels,'Standardize', true, 'KernelFunction','rbf',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)

% % Best Observed - BC: 582.15, KS: 154.85
% % Best Estimate - BC: 255.95, KS: 141.74

% Training
classifier = fitcsvm(trainingFeatures,imgTrain.Labels,'KernelFunction','linear','Standardize',true, ...
  'ClassNames',{'N5' '5'},'BoxConstraint', 255.95, 'KernelScale', 141.74);


%% Classify Test Images
featureLayer = 'fc7';
testFeatures = activations(net, augmentedTestImages, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
[predictedLabels, scores] = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

finalScores(:,4) = scores(:,2);

%% Determine Accuracy
fivesAccuracy = getTestAccuracy(predictedLabels, imgTestLabels)








%% 6's:
%% CNN Feature Extraction :
label = '6';
[files, labels] = getTrainingDataset(trainImages, label, allLabels);
imgTrain = imageDatastore(...
    files, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');
imgTrain.Labels = labels;

imgTrain.Labels = updateImgLabels(imgTrain, label);
imgTestLabels = updateImgLabels(testImages, label);

[trainingFeatures, net] = getSuitFeatures(imgTrain);

%% Optimize Hyperparameters

% Change # to size of training data passed in
% c = cvpartition(length(imgTrain.Labels),'KFold',10);
% opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
%     'AcquisitionFunctionName','expected-improvement-plus');
% SVMModel = fitcsvm(trainingFeatures,imgTrain.Labels,'Standardize', true, 'KernelFunction','rbf',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)

% % Best Observed - BC: 394.51, KS: 153.34
% % Best Estimate - BC: 401.54, KS: 956.44

% Training
classifier = fitcsvm(trainingFeatures,imgTrain.Labels,'KernelFunction','linear','Standardize',true, ...
  'ClassNames',{'N6' '6'},'BoxConstraint', 401.54, 'KernelScale', 956.44);


%% Classify Test Images
featureLayer = 'fc7';
testFeatures = activations(net, augmentedTestImages, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
[predictedLabels, scores] = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

finalScores(:,5) = scores(:,2);

%% Determine Accuracy
sixesAccuracy = getTestAccuracy(predictedLabels, imgTestLabels)








%% 7's:
%% CNN Feature Extraction :
label = '7';
[files, labels] = getTrainingDataset(trainImages, label, allLabels);
imgTrain = imageDatastore(...
    files, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');
imgTrain.Labels = labels;

imgTrain.Labels = updateImgLabels(imgTrain, label);
imgTestLabels = updateImgLabels(testImages, label);

[trainingFeatures, net] = getSuitFeatures(imgTrain);

%% Optimize Hyperparameters

% Change # to size of training data passed in
% c = cvpartition(length(imgTrain.Labels),'KFold',10);
% opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
%     'AcquisitionFunctionName','expected-improvement-plus');
% SVMModel = fitcsvm(trainingFeatures,imgTrain.Labels,'Standardize', true, 'KernelFunction','rbf',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)

% % Best Observed - BC: 992.11, KS: 548.26
% % Best Estimate - BC: 960.17, KS: 961.6

% Training
classifier = fitcsvm(trainingFeatures,imgTrain.Labels,'KernelFunction','linear','Standardize',true, ...
  'ClassNames',{'N7' '7'},'BoxConstraint', 960.17, 'KernelScale', 961.6);


%% Classify Test Images
featureLayer = 'fc7';
testFeatures = activations(net, augmentedTestImages, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
[predictedLabels, scores] = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

finalScores(:,6) = scores(:,2);

%% Determine Accuracy
sevensAccuracy = getTestAccuracy(predictedLabels, imgTestLabels)








%% 8's:
%% CNN Feature Extraction :
label = '8';
[files, labels] = getTrainingDataset(trainImages, label, allLabels);
imgTrain = imageDatastore(...
    files, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');
imgTrain.Labels = labels;

imgTrain.Labels = updateImgLabels(imgTrain, label);
imgTestLabels = updateImgLabels(testImages, label);

[trainingFeatures, net] = getSuitFeatures(imgTrain);

%% Optimize Hyperparameters

% Change # to size of training data passed in
% c = cvpartition(length(imgTrain.Labels),'KFold',10);
% opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
%     'AcquisitionFunctionName','expected-improvement-plus');
% SVMModel = fitcsvm(trainingFeatures,imgTrain.Labels,'Standardize', true, 'KernelFunction','rbf',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)

% % Best Observed - BC: 969.9, KS: 537.44
% % Best Estimate - BC: 978.11, KS: 487.26

% Training
classifier = fitcsvm(trainingFeatures,imgTrain.Labels,'KernelFunction','linear','Standardize',true, ...
  'ClassNames',{'N8' '8'},'BoxConstraint', 978.11, 'KernelScale', 487.26);


%% Classify Test Images
featureLayer = 'fc7';
testFeatures = activations(net, augmentedTestImages, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
[predictedLabels, scores] = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

finalScores(:,7) = scores(:,2);

%% Determine Accuracy
eightsAccuracy = getTestAccuracy(predictedLabels, imgTestLabels)









%% 9's:
%% CNN Feature Extraction :
label = '9';
[files, labels] = getTrainingDataset(trainImages, label, allLabels);
imgTrain = imageDatastore(...
    files, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');
imgTrain.Labels = labels;

imgTrain.Labels = updateImgLabels(imgTrain, label);
imgTestLabels = updateImgLabels(testImages, label);

[trainingFeatures, net] = getSuitFeatures(imgTrain);

%% Optimize Hyperparameters

% Change # to size of training data passed in
% c = cvpartition(length(imgTrain.Labels),'KFold',10);
% opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
%     'AcquisitionFunctionName','expected-improvement-plus');
% SVMModel = fitcsvm(trainingFeatures,imgTrain.Labels,'Standardize', true, 'KernelFunction','rbf',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)

% % Best Observed - BC: 223.14, KS: 363.14
% % Best Estimate - BC: 990.32, KS: 96.259

% Training
classifier = fitcsvm(trainingFeatures,imgTrain.Labels,'KernelFunction','linear','Standardize',true, ...
  'ClassNames',{'N9' '9'},'BoxConstraint', 990.32, 'KernelScale', 96.259);


%% Classify Test Images
featureLayer = 'fc7';
testFeatures = activations(net, augmentedTestImages, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
[predictedLabels, scores] = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

finalScores(:,8) = scores(:,2);

%% Determine Accuracy
ninesAccuracy = getTestAccuracy(predictedLabels, imgTestLabels)









%% 10's:
%% CNN Feature Extraction :
label = '10';
[files, labels] = getTrainingDataset(trainImages, label, allLabels);
imgTrain = imageDatastore(...
    files, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');
imgTrain.Labels = labels;

imgTrain.Labels = updateImgLabels(imgTrain, label);
imgTestLabels = updateImgLabels(testImages, label);

[trainingFeatures, net] = getSuitFeatures(imgTrain);

%% Optimize Hyperparameters

% Change # to size of training data passed in
% c = cvpartition(length(imgTrain.Labels),'KFold',10);
% opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
%     'AcquisitionFunctionName','expected-improvement-plus');
% SVMModel = fitcsvm(trainingFeatures,imgTrain.Labels,'Standardize', true, 'KernelFunction','rbf',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)

% % Best Observed - BC: 344.8, KS: 108.96
% % Best Estimate - BC: 991.8, KS: 127.91

% Training
classifier = fitcsvm(trainingFeatures,imgTrain.Labels,'KernelFunction','linear','Standardize',true, ...
  'ClassNames',{'N10' '10'},'BoxConstraint', 991.8, 'KernelScale', 127.91);


%% Classify Test Images
featureLayer = 'fc7';
testFeatures = activations(net, augmentedTestImages, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
[predictedLabels, scores] = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

finalScores(:,9) = scores(:,2);

%% Determine Accuracy
tensAccuracy = getTestAccuracy(predictedLabels, imgTestLabels)






%% A's:
%% CNN Feature Extraction :
label = 'A';
[files, labels] = getTrainingDataset(trainImages, label, allLabels);
imgTrain = imageDatastore(...
    files, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');
imgTrain.Labels = labels;

imgTrain.Labels = updateImgLabels(imgTrain, label);
imgTestLabels = updateImgLabels(testImages, label);

[trainingFeatures, net] = getSuitFeatures(imgTrain);

%% Optimize Hyperparameters

% Change # to size of training data passed in
% c = cvpartition(length(imgTrain.Labels),'KFold',10);
% opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
%     'AcquisitionFunctionName','expected-improvement-plus');
% SVMModel = fitcsvm(trainingFeatures,imgTrain.Labels,'Standardize', true, 'KernelFunction','rbf',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)

% % Best Observed - BC: 746.43, KS: 109.86
% % Best Estimate - BC: 746.43, KS: 109.86

% Training
classifier = fitcsvm(trainingFeatures,imgTrain.Labels,'KernelFunction','linear','Standardize',true, ...
  'ClassNames',{'NA' 'A'},'BoxConstraint', 746.43, 'KernelScale', 109.86);


%% Classify Test Images
featureLayer = 'fc7';
testFeatures = activations(net, augmentedTestImages, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
[predictedLabels, scores] = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

finalScores(:,10) = scores(:,2);

%% Determine Accuracy
acesAccuracy = getTestAccuracy(predictedLabels, imgTestLabels)







%% J's:
%% CNN Feature Extraction :
label = 'J';
[files, labels] = getTrainingDataset(trainImages, label, allLabels);
imgTrain = imageDatastore(...
    files, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');
imgTrain.Labels = labels;

imgTrain.Labels = updateImgLabels(imgTrain, label);
imgTestLabels = updateImgLabels(testImages, label);

[trainingFeatures, net] = getSuitFeatures(imgTrain);

%% Optimize Hyperparameters

% Change # to size of training data passed in
% c = cvpartition(length(imgTrain.Labels),'KFold',10);
% opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
%     'AcquisitionFunctionName','expected-improvement-plus');
% SVMModel = fitcsvm(trainingFeatures,imgTrain.Labels,'Standardize', true, 'KernelFunction','rbf',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)

% % Best Observed - BC: 404.83, KS: 358.1
% % Best Estimate - BC: 520.57, KS: 340.28

% Training
classifier = fitcsvm(trainingFeatures,imgTrain.Labels,'KernelFunction','linear','Standardize',true, ...
  'ClassNames',{'NJ' 'J'},'BoxConstraint', 520.57, 'KernelScale', 340.28);


%% Classify Test Images
featureLayer = 'fc7';
testFeatures = activations(net, augmentedTestImages, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
[predictedLabels, scores] = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

finalScores(:,11) = scores(:,2);

%% Determine Accuracy
jacksAccuracy = getTestAccuracy(predictedLabels, imgTestLabels)




%% Q's:
%% CNN Feature Extraction :
label = 'Q';
[files, labels] = getTrainingDataset(trainImages, label, allLabels);
imgTrain = imageDatastore(...
    files, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');
imgTrain.Labels = labels;

imgTrain.Labels = updateImgLabels(imgTrain, label);
imgTestLabels = updateImgLabels(testImages, label);

[trainingFeatures, net] = getSuitFeatures(imgTrain);

%% Optimize Hyperparameters

% Change # to size of training data passed in
% c = cvpartition(length(imgTrain.Labels),'KFold',10);
% opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
%     'AcquisitionFunctionName','expected-improvement-plus');
% SVMModel = fitcsvm(trainingFeatures,imgTrain.Labels,'Standardize', true, 'KernelFunction','rbf',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)

% % Best Observed - BC: 301.01, KS: 963.16
% % Best Estimate - BC: 348.45, KS: 999.81

% Training
classifier = fitcsvm(trainingFeatures,imgTrain.Labels,'KernelFunction','linear','Standardize',true, ...
  'ClassNames',{'NQ' 'Q'},'BoxConstraint', 348.45, 'KernelScale', 999.81);


%% Classify Test Images
featureLayer = 'fc7';
testFeatures = activations(net, augmentedTestImages, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
[predictedLabels, scores] = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

finalScores(:,12) = scores(:,2);

%% Determine Accuracy
queensAccuracy = getTestAccuracy(predictedLabels, imgTestLabels)







%% K's:
%% CNN Feature Extraction :
label = 'K';
[files, labels] = getTrainingDataset(trainImages, label, allLabels);
imgTrain = imageDatastore(...
    files, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');
imgTrain.Labels = labels;

imgTrain.Labels = updateImgLabels(imgTrain, label);
imgTestLabels = updateImgLabels(testImages, label);

[trainingFeatures, net] = getSuitFeatures(imgTrain);

%% Optimize Hyperparameters

% Change # to size of training data passed in
% c = cvpartition(length(imgTrain.Labels),'KFold',10);
% opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
%     'AcquisitionFunctionName','expected-improvement-plus');
% SVMModel = fitcsvm(trainingFeatures,imgTrain.Labels,'Standardize', true, 'KernelFunction','rbf',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)

% % Best Observed - BC: 906.37, KS: 103.17
% % Best Estimate - BC: 491.1, KS: 100.91

% Training
classifier = fitcsvm(trainingFeatures,imgTrain.Labels,'KernelFunction','linear','Standardize',true, ...
  'ClassNames',{'NK' 'K'},'BoxConstraint', 491.1, 'KernelScale', 100.91);


%% Classify Test Images
featureLayer = 'fc7';
testFeatures = activations(net, augmentedTestImages, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
[predictedLabels, scores] = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

finalScores(:,13) = scores(:,2);

%% Determine Accuracy
kingsAccuracy = getTestAccuracy(predictedLabels, imgTestLabels)









%% OVERALL - Determine Most Likely Rank

% Determine most likely rank from scores
maxScores = max(finalScores, [], 2);
bestRank = cell(1,length(maxScores));
for i = 1:length(maxScores)
    idx = find(finalScores(i,:) == maxScores(i));

    switch idx
        case 1
            bestRank(i) = cellstr("2");
        case 2
            bestRank(i) = cellstr("3");
        case 3
            bestRank(i) = cellstr("4");
        case 4
            bestRank(i) = cellstr("5");
        case 5
            bestRank(i) = cellstr("6");
        case 6
            bestRank(i) = cellstr("7");
        case 7
            bestRank(i) = cellstr("8");
        case 8
            bestRank(i) = cellstr("9");
        case 9
            bestRank(i) = cellstr("10");
        case 10
            bestRank(i) = cellstr("A");
        case 11
            bestRank(i) = cellstr("J");
        case 12
            bestRank(i) = cellstr("Q");
        case 13
            bestRank(i) = cellstr("K");
    end
end

% Determine Overall Accuracy
count = 0;
for i = 1:length(bestRank)
   if contains(cellstr(testImages.Labels(i)), bestRank(i))
       count = count + 1;
   end
end
overallAccuracy = count/length(bestRank)

finalLabels = [testImages.Labels(:) categorical(bestRank') testImages.Files(:)];

rank = finalLabels;

end