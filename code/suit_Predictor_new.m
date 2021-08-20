clc;
clear all;
rootDir = 'C:\Users\brownjw1\Documents\CSSE_463_Image Recognition\Final_Project\CSSE-463-Poker-Player\';
trainDir = [rootDir '\dataset\custom_train\suit'];
trainImages = imageDatastore(...
   trainDir, ...
   'IncludeSubfolders',true, ...
   'LabelSource', 'foldernames');

 

% Create Transfer Network
network = alexnet;
inputSize = network.Layers(1).InputSize;

transferredLayers = network.Layers(1:end-3);

[imgTrain,imgValidation] = splitEachLabel(trainImages,0.7,'randomized');

numClasses = numel(unique(trainImages.Labels));

layers = [
  transferredLayers
  fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 30, 'BiasLearnRateFactor', 20)
  softmaxLayer
  classificationLayer];

augmentedTrainingImages = augmentedImageDatastore(inputSize(1:2),imgTrain,'ColorPreprocessing', 'gray2rgb');
augmentedValidationImages = augmentedImageDatastore(inputSize(1:2),imgValidation,'ColorPreprocessing', 'gray2rgb');
 
% Train Model
options = trainingOptions('adam', ...
    'MiniBatchSize',20, ...
    'MaxEpochs',6, ...
    'ValidationPatience',10, ...
    'ValidationFrequency',3, ...
    'Plots','training-progress', ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'ValidationData',augmentedValidationImages, ...
    'Plots','training-progress');

transferNetwork = trainNetwork(augmentedTrainingImages, layers, options);

% Test Data
testImages = imageDatastore(...
   strcat(rootDir, '\dataset\custom_test\suit'), ...
   'IncludeSubfolders',true, ...
   'LabelSource', 'foldernames');
augmentedTestImages = augmentedImageDatastore(inputSize(1:2),testImages, 'ColorPreprocessing', 'gray2rgb');

[YPred,scores] = classify(transferNetwork,augmentedTestImages);

matches = find(YPred == testImages.Labels);
matches = YPred(matches);

accuracy = length(matches)/length(YPred)
save 'Ypred_Suits.mat' YPred
testLabels = testImages.Labels;
save 'YTest_Ranks.mat' testLabels
testLabels = testImages.Labels;
save 'YTest_Suits.mat' testLabels

