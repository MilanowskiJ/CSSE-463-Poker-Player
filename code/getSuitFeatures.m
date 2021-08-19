function [trainingFeatures, net] = getSuitFeatures(imageDatastore)
%getSuitFeatures
% Extracts features from the input image dataset.  Initializes Alexnet,
% trains the network, extracts features from the 'fc7' layer, and returns
% them as well as the set of Test Images.

net = alexnet;
inputSize = net.Layers(1).InputSize;

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange);

augmentedTrainingImages = augmentedImageDatastore(inputSize(1:2),imageDatastore,'ColorPreprocessing', 'gray2rgb', 'DataAugmentation',imageAugmenter);

featureLayer = 'fc7';
trainingFeatures = activations(net, augmentedTrainingImages, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Fix the Dimensions of the Feature Matrix
trainingFeatures = trainingFeatures';

end

