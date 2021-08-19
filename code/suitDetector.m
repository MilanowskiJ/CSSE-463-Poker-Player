
function [suit] = suitDetector()

%% Reset and Read in Images
% clear all; close all; clc;

% Set Directory Paths
rootDir = 'C:\Users\boewebe\Documents\CSSE463-ImageRecognition\Project\8-13\CSSE-463-Poker-Player-main';
trainDir = [rootDir '\dataset\custom_train\suit'];
testDir = [rootDir '\dataset\custom_test\suit'];

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
finalScores = zeros(length(testImages.Labels), 4);
allLabels = ['C'; 'S'; 'H'; 'D'];




%% CLUBS:
%% CNN Feature Extraction :
label = 'C';
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

% % Best Observed - BC: 38.229, KS: 81.44
% % Best Estimate - BC: 278.66, KS: 60.462

% Training
classifier = fitcsvm(trainingFeatures,imgTrain.Labels,'KernelFunction','linear','Standardize',true, ...
  'ClassNames',{'NC' 'C'}, 'BoxConstraint', 278.66, 'KernelScale', 60.462);


%% Classify Test Images
featureLayer = 'fc7';
testFeatures = activations(net, augmentedTestImages, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
[predictedLabels, scores] = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

finalScores(:,1) = scores(:,2);

%% Determine Accuracy
clubsAccuracy = getTestAccuracy(predictedLabels, imgTestLabels)





% Same process is used for all other suits below







%% SPADES:
%% CNN Feature Extraction :
label = 'S';
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

% % Best Observed - BC: 371, KS: 86.124
% % Best Estimate - BC: 214.96, KS: 80.947

% Training
classifier = fitcsvm(trainingFeatures,imgTrain.Labels,'KernelFunction','linear','Standardize',true, ...
  'ClassNames',{'NS' 'S'}, 'BoxConstraint', 214.96, 'KernelScale', 80.947);


%% Classify Test Images
featureLayer = 'fc7';
testFeatures = activations(net, augmentedTestImages, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
[predictedLabels, scores] = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

finalScores(:,2) = scores(:,2);

%% Determine Accuracy
spadesAccuracy = getTestAccuracy(predictedLabels, imgTestLabels)


















%% HEARTS:
%% CNN Feature Extraction :
label = 'H';
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

% % Best Observed - BC: 1.756, KS: 72.082
% % Best Estimate - BC: 432.17, KS: 103.2

% Training
classifier = fitcsvm(trainingFeatures,imgTrain.Labels,'KernelFunction','linear','Standardize',true, ...
  'ClassNames',{'NH' 'H'},'BoxConstraint', 432.17, 'KernelScale', 103.2);


%% Classify Test Images
featureLayer = 'fc7';
testFeatures = activations(net, augmentedTestImages, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
[predictedLabels, scores] = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

finalScores(:,3) = scores(:,2);

%% Determine Accuracy
heartsAccuracy = getTestAccuracy(predictedLabels, imgTestLabels)













%% DIAMONDS:
%% CNN Feature Extraction :
label = 'D';
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

% % Best Observed - BC: 952.55, KS: 154.64
% % Best Estimate - BC: 986.45, KS: 160.08

% Training
classifier = fitcsvm(trainingFeatures,imgTrain.Labels,'KernelFunction','linear','Standardize',true, ...
  'ClassNames',{'ND' 'D'},'BoxConstraint', 986.45, 'KernelScale', 160.08);


%% Classify Test Images
featureLayer = 'fc7';
testFeatures = activations(net, augmentedTestImages, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
[predictedLabels, scores] = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

finalScores(:,4) = scores(:,2);

%% Determine Accuracy
diamondsAccuracy = getTestAccuracy(predictedLabels, imgTestLabels)







%% OVERALL - Determine Most Likely Suit

% Determine most likely suit from scores
maxScores = max(finalScores, [], 2);
bestSuit = cell(1,length(maxScores));
for i = 1:length(maxScores)
    idx = find(finalScores(i,:) == maxScores(i));

    switch idx
        case 1
            bestSuit(i) = cellstr("C");
        case 2
            bestSuit(i) = cellstr("S");
        case 3
            bestSuit(i) = cellstr("H");
        case 4
            bestSuit(i) = cellstr("D");
    end
end

% Determine Overall Accuracy
count = 0;
for i = 1:length(bestSuit)
   if contains(cellstr(testImages.Labels(i)), bestSuit(i))
       count = count + 1;
   end
end
overallAccuracy = count/length(bestSuit)

finalLabels = [testImages.Labels(:) categorical(bestSuit') testImages.Files(:)];

suit = finalLabels;
end