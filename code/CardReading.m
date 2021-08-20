%% Clear variables and console <LOCKED>
clear all; clc;

%% Declare network skeleton <LOCKED>
net = squeezenet;
inputSize = [920 968 3];

%% Load ground truth card readings <VARIABLE: gTruthData>
gTruthData = 'cornerColorCardTrainingSetGroundTruth';
load(strcat(gTruthData,'.mat'));
files = table(gTruth.DataSource.Source,'VariableNames',{'imageFilename'});
TrainingCards = [files, gTruth.LabelData];

%% Split TrainingCards {100% Train - 0% Validation - 0% Test} <LOCKED>
rng(0) % rng seed
shuffledIndices = randperm(height(TrainingCards));
idx = height(TrainingCards);

trainingIdx = 1:idx;
trainingDataTbl = TrainingCards(shuffledIndices(trainingIdx),:);

%% Convert datasets to imageDatastore and boxLabelDatastore <LOCKED>
imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'corner'));

%% Combine datastores <LOCKED>
trainingData = combine(imdsTrain,bldsTrain);

%% Display an example image with bounding boxes <LOCKED>
data = read(trainingData);
I = data{1}; bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure; imshow(annotatedImage);

%% Checks the layout of the network <LOCKED>
if isa(net,'SeriesNetwork')
    lgraph = layerGraph(net.Layers);
else
    lgraph = layerGraph(net);
end

%% Find and replace the last two layers <LOCKED>
[learnableLayer,classLayer] = findLayersToReplace(lgraph);

numClasses = 2;

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

%% Freeze the beginning layers <LOCKED>
layers = lgraph.Layers;
connections = lgraph.Connections;

%layers(1:60) = freezeWeights(layers(1:60));
lgraph = createLgraphUsingConnections(layers,connections);

%% Train data augmentation before network training <LOCKED>
trainingData = transform(trainingData,@(data)preprocessData(data,inputSize));
augmentedData = cell(4,1);
for k = 1:randi(size(trainingData,1),1,4)
    data = read(trainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(trainingData);
end
figure; montage(augmentedData,'BorderSize',10);

%% Display an example image with bounding boxes <LOCKED>
data = read(trainingData);
I = data{1}; bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure; imshow(annotatedImage);

%% Set options and train the network <VARIABLE: MiniBatchSize, MaxEpochs>
% Set network training options to use mini-batch size of 32 to reduce GPU
% memory usage. Lower the InitialLearnRate to reduce the rate at which
% network parameters are changed. This is beneficial when fine-tuning a
% pre-trained network and prevents the network from changing too rapidly.
miniBatchSize = 10; maxEpochs = 5;
options = trainingOptions('sgdm', ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', 1e-6, ...
    'MaxEpochs', maxEpochs, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress');

%% Train <VARIABLE: negR, posR>
negR = [0 0.1]; posR = [0.7 1];
inputTraining = table(trainingData.UnderlyingDatastores{1}.UnderlyingDatastores{1}.Files, ...
      {trainingData.UnderlyingDatastores{1}.UnderlyingDatastores{2}.LabelData{:,1}}','VariableNames',{'imageFilename','corner'});
rcnn = trainFastRCNNObjectDetector(inputTraining, lgraph, options, ...
       'NegativeOverlapRange', negR, 'PositiveOverlapRange', posR, 'FreezeBatchNormalization', true);
       %'BoxRegressionLayer', 'new_classoutput');

%% Save detected corners in test dataset <VARIABLE: detectedColorGrayTestFolder, testFolder>
detectedColorGrayTestFolder = 'detectedColorTest1'; testFolder = 'test';
if isequal(testFolder, 'grayTest')
    x = 111;
elseif isequal(testFolder, 'test')
    x = 107;
end

rootdir = '/Users/sirigecj/Desktop/Rose_Hulman/Sophomore/Sophomore Summer/Term Project/PlayingCardReaderFolder/';
testImages = imageDatastore(...
                            [rootdir testFolder], ...
                            'IncludeSubfolders',true, ...
                            'LabelSource', 'foldernames');

for k = 1:size(testImages.Files,1)
    I = imread(testImages.Files{k});
    I = imresize(I,inputSize(1:2));
    [bboxes,scores] = detect(rcnn,I);
    if size(bboxes,1) ~= 0
        I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
    end
    imshow(I);
    imwrite(I,strcat(detectedColorGrayTestFolder,'\',extractAfter(testImages.Files{k},x)));
end
matFile = sprintf('fastRCNN_%s_%s_%s_%s_%s_%s_%s.mat', gTruthData, num2str(miniBatchSize), num2str(maxEpochs), ...
          extractAfter(num2str(negR),2), extractAfter(num2str(posR),2), testFolder, detectedColorGrayTestFolder);
save(matFile,'rcnn');

%% Classify cards
% % Clear variables and console <LOCKED>
% clear all; clc;
% 
% % Declare network skeleton <LOCKED>
% net = squeezenet;
% inputSize = net.Layers(1).InputSize;
% 
% % Load ground truth card readings <VARIABLE: gTruthData>
% gTruthData = 'cornerColorCardTrainingSetGroundTruth';
% load(strcat(gTruthData,'.mat'));
% files = table(gTruth.DataSource.Source,'VariableNames',{'imageFilename'});
% TrainingCards = [files, gTruth.LabelData];
% 
% % Split TrainingCards {92% Train - 0% Validation - 8% Test} <LOCKED>
% rng(0) % rng seed
% shuffledIndices = randperm(height(TrainingCards));
% idx = floor(0.92 * height(TrainingCards));
% 
% trainingIdx = 1:idx;
% trainingDataTbl = TrainingCards(shuffledIndices(trainingIdx),:);
% 
% testIdx = idx+1 : length(shuffledIndices);
% testDataTbl = TrainingCards(shuffledIndices(testIdx),:);
% 
% load('similarToFCNN_70Train_10Val_20Test_squeezenet_unFrozen_32_10_shuffled_40_75_color1.mat');
% 
% %% Part of classify cards
% figure;
% for k = 1:size(testDataTbl.imageFilename,1)
%     I = imread(testDataTbl.imageFilename{k});
%     I = imresize(I,inputSize(1:2));
%     [bboxes,scores] = detect(rcnn,I);
%     if size(bboxes,1) ~= 0
%         I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
%         imshow(I);
%         for j = 1:size(bboxes,1)
%             Icropped = imcrop(I,bboxes(j,:));
%             figure; imshow(Icropped);
%             BW = ~imbinarize(im2gray(Icropped)); imshow(BW);
%             CC = bwconncomp(BW,8);
%             stats = regionprops(CC,'Centroid');
%         end
%     end
% end
% 
% rootdir = '/Users/sirigecj/Desktop/Rose_Hulman/Sophomore/Sophomore Summer/Term Project/PlayingCardReaderFolder/';
% testImages = imageDatastore(...
%                             [rootdir 'test'], ...
%                             'IncludeSubfolders',true, ...
%                             'LabelSource', 'foldernames');
% %
% figure;
% for k = 1:size(testImages.Files,1)
%     I = imread(testImages.Files{k});
%     I = imresize(I,inputSize(1:2));
%     [bboxes,scores] = detect(rcnn,I);
%     if size(bboxes,1) ~= 0
%         I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
%         imshow(I);
%         for j = 1:size(bboxes,1)
%             Icropped = imcrop(I,bboxes(j,:));
%             figure; imshow(Icropped);
%             BW = ~imbinarize(im2gray(Icropped)); imshow(BW);
%             CC = bwconncomp(BW,8);
%             stats = regionprops(CC,'Centroid');
%         end
%     end
% end

%% Stuff
% %% Evaluate Detector on entire test set
% testData = transform(testData,@(data)preprocessData(data,inputSize));
% detectionResults = detect(rcnn,testData,'MinibatchSize',miniBatchSize);
% [ap, recall, precision] = evaluateDetectionPrecision(detectionResults,testData);
% 
% %% Display precision/recall (PR) curve
% figure; plot(recall,precision)
% xlabel('Recall'); ylabel('Precision');
% grid on;
% title(sprintf('Average Precision = %.2f', ap));

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

function data = preprocessData(data,targetSize)
    % Resize image and bounding boxes to the targetSize.
    scale = targetSize(1:2)./size(data{1},[1 2]);
    data{1} = imresize(data{1},targetSize(1:2));
    bboxes = round(data{2});
    data{2} = bboxresize(bboxes,scale);
end