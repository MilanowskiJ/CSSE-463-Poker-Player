function [SuitPredictions,SuitTest] = suit_Predictor_crop(imgTrain, imgValidation, imgTest)
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

[YPred,scores] = classify(transferNetwork,augmentedTestImages);

save 'croppedSuitNetwork.mat' transferNetwork;
matches = find(YPred == imgTest.Labels);
matches = YPred(matches);

accuracy = length(matches)/length(YPred);

figure(2)
confusionchart(imgTest.Labels, YPred);
fprintf('Suit Prediction Accuracy: %0.2f\n' , accuracy);
SuitPredictions = YPred;
SuitTest = imgTest.Labels;

end
