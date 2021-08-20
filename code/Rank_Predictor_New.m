% Note: Must run suit predictor first.


clc;
clear all;
rootDir = 'C:\Users\brownjw1\Documents\CSSE_463_Image Recognition\Final_Project\CSSE-463-Poker-Player\';
trainDir = [rootDir '\dataset\custom_train\rank'];

trainImages = imageDatastore(...
   trainDir, ...
   'IncludeSubfolders',true, ...
   'LabelSource', 'foldernames');

% correctImageRotation(trainImages.

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
    'MiniBatchSize',30, ...
    'MaxEpochs',7, ...
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
   strcat(rootDir, '\dataset\custom_test\rank'), ...
   'IncludeSubfolders',true, ...
   'LabelSource', 'foldernames');
augmentedTestImages = augmentedImageDatastore(inputSize(1:2),testImages, 'ColorPreprocessing', 'gray2rgb');

[YPredRanks,scores] = classify(transferNetwork,augmentedTestImages);
matches = find(YPredRanks == testImages.Labels);
matches = YPredRanks(matches);

accuracy = length(matches)/length(YPredRanks)

YPredSuits = load('Ypred_Suits.mat');
Predictions = horzcat(YPredRanks,YPredSuits.YPred)


for c= 1:length(Predictions)
    FinalPredictions(c,1) = strcat(string(Predictions(c,1)), string(Predictions(c,2)));
end

TestLabels = load('YTest_Suits.mat')

TestLabelSuits = TestLabels.testLabels;
TestLabelRanks = testImages.Labels;

for k= 1:length(TestLabelSuits)
    FinalTestLabels(k,1) = strcat(string(TestLabelRanks(k)), string(TestLabelSuits(k)));
end


matches2 = find(FinalPredictions == FinalTestLabels);
matches2 = FinalPredictions(matches2);
accuracy = length(matches2)/length(FinalPredictions)


save 'FinalPredictionsNoRotation.mat' FinalPredictions



