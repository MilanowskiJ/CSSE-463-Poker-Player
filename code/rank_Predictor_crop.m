function [RankPredictions,RankTest] = rank_Predictor_crop(imgTrain, imgValidation, imgTest)

% Create Transfer Network
network = alexnet;
inputSize = network.Layers(1).InputSize;

transferredLayers = network.Layers(1:end-3);

numClasses = numel(unique(imgTrain.Labels));

layers = [
  transferredLayers
  fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 30, 'BiasLearnRateFactor', 20)
  softmaxLayer
  classificationLayer];

augmentedTrainingImages = augmentedImageDatastore(inputSize(1:2),imgTrain,'ColorPreprocessing', 'gray2rgb');
augmentedValidationImages = augmentedImageDatastore(inputSize(1:2),imgValidation,'ColorPreprocessing', 'gray2rgb');
augmentedTestImages = augmentedImageDatastore(inputSize(1:2),imgTest,'ColorPreprocessing', 'gray2rgb');

% Train Model
options = trainingOptions('adam', ...
    'MiniBatchSize',100, ...
    'MaxEpochs',4, ...
    'ValidationPatience',10, ...
    'ValidationFrequency',3, ...
    'Plots','training-progress', ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'ValidationData',augmentedValidationImages, ...
    'Plots','training-progress');

transferNetwork = trainNetwork(augmentedTrainingImages, layers, options);
save 'croppedRankNetwork.mat' transferNetwork;

[YPredRanks,scores] = classify(transferNetwork,augmentedTestImages);
matches = find(YPredRanks == imgTest.Labels);
matches = YPredRanks(matches);

accuracy = length(matches)/length(YPredRanks);
fprintf('Rank Prediction Accuracy: %0.2f\n' , accuracy);

RankPredictions = YPredRanks;
RankTest = imgTest.Labels;
figure(3)
confusionchart(RankTest, RankPredictions);

end



