%% Clear variables and console
clear all; clc;

%% Load ground truth card readings
load('cornerColorCardTrainingSetGroundTruth.mat')
files = table(gTruth.DataSource.Source,'VariableNames',{'imageFilename'});
TrainingCards = [files,gTruth.LabelData];

%% Declare network skeleton
net = squeezenet;
inputSize = net.Layers(1).InputSize;

%% Read images using Datastore and rescale
rootdir = '/Users/sirigecj/Desktop/Rose_Hulman/Sophomore/Sophomore Summer/Term Project/PlayingCardReaderFolder/';

% Train image
%trainImages = imageDatastore(...
%    [rootdir 'train'], ...
%    'ReadFcn', @rescale, ...
%    'IncludeSubfolders',true, ...
%    'LabelSource', 'foldernames');

% Test image set    %'ReadFcn', @rescale, ...
testImages = imageDatastore(...
    [rootdir 'test'], ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

%% Augment the images
%augmentedTrainImages = augmentedImageDatastore(inputSize(1:2),trainImages,'ColorPreprocessing','rgb2gray');
%augmentedTestImages = augmentedImageDatastore(inputSize(1:2),testImages,'ColorPreprocessing','rgb2gray');

%% Checks the layout of the network
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers);
else
  lgraph = layerGraph(net);
end

%% Find and replace the last two layers
[learnableLayer,classLayer] = findLayersToReplace(lgraph);

numClasses = (size(TrainingCards,2) - 1) + 1;

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

%% Freeze the beginning layers
layers = lgraph.Layers;
connections = lgraph.Connections;

%layers(1:60) = freezeWeights(layers(1:60));
lgraph = createLgraphUsingConnections(layers,connections);

%% Set options and train the network
% Set network training options to use mini-batch size of 32 to reduce GPU
% memory usage. Lower the InitialLearnRate to reduce the rate at which
% network parameters are changed. This is beneficial when fine-tuning a
% pre-trained network and prevents the network from changing too rapidly.
miniBatchSize = 32;
options = trainingOptions('sgdm', ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', 1e-6, ...
    'MaxEpochs', 10, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress');

%% Train
% Train the R-CNN detector. Training can take a few minutes to complete.
rcnn = trainRCNNObjectDetector(TrainingCards, lgraph, options, ...
       'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.6 1], ...
       'BoxRegressionLayer', 'new_classoutput');

%%
% Test the R-CNN detector on a test image.
img = imread(testImages.Files{1});

[bbox, score, label] = detect(rcnn, img, 'MiniBatchSize', miniBatchSize);

% Display strongest detection result.
[score, idx] = max(score);

bbox = bbox(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, annotation);

figure
imshow(detectedImg)

%% Additional functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [learnableLayer,classLayer] = findLayersToReplace(lgraph)
    if ~isa(lgraph,'nnet.cnn.LayerGraph')
        error('Argument must be a LayerGraph object.')
    end

    % Get source, destination, and layer names.
    src = string(lgraph.Connections.Source);
    dst = string(lgraph.Connections.Destination);
    layerNames = string({lgraph.Layers.Name}');

    % Find the classification layer. The layer graph must have a single
    % classification layer.
    isClassificationLayer = arrayfun(@(l) ...
        (isa(l,'nnet.cnn.layer.ClassificationOutputLayer')|isa(l,'nnet.layer.ClassificationLayer')), ...
        lgraph.Layers);

    if sum(isClassificationLayer) ~= 1
        error('Layer graph must have a single classification layer.')
    end
    classLayer = lgraph.Layers(isClassificationLayer);

    % Traverse the layer graph in reverse starting from the classification
    % layer. If the network branches, throw an error.
    currentLayerIdx = find(isClassificationLayer);
    while true
        if numel(currentLayerIdx) ~= 1
            error('Layer graph must have a single learnable layer preceding the classification layer.')
        end

        currentLayerType = class(lgraph.Layers(currentLayerIdx));
        isLearnableLayer = ismember(currentLayerType, ...
            ['nnet.cnn.layer.FullyConnectedLayer','nnet.cnn.layer.Convolution2DLayer']);

        if isLearnableLayer
            learnableLayer =  lgraph.Layers(currentLayerIdx);
            return
        end

        currentDstIdx = find(layerNames(currentLayerIdx) == dst);
        currentLayerIdx = find(src(currentDstIdx) == layerNames);
    end
end

% layers = freezeWeights(layers) sets the learning rates of all the
% parameters of the layers in the layer array |layers| to zero.
function layers = freezeWeights(layers)
    for ii = 1:size(layers,1)
        props = properties(layers(ii));
        for p = 1:numel(props)
            propName = props{p};
            if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
                layers(ii).(propName) = 0;
            end
        end
    end

end

% lgraph = createLgraphUsingConnections(layers,connections) creates a layer
% graph with the layers in the layer array |layers| connected by the
% connections in |connections|.
function lgraph = createLgraphUsingConnections(layers,connections)
    lgraph = layerGraph();
    for i = 1:numel(layers)
        lgraph = addLayers(lgraph,layers(i));
    end

    for c = 1:size(connections,1)
        lgraph = connectLayers(lgraph,connections.Source{c},connections.Destination{c});
    end

end