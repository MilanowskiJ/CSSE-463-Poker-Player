%% Clear variables and console
clear all; clc; gpu = gpuDevice(); reset(gpu);
fprintf('\tClear variables and console\n');

%% Load ground truth card readings

            tStart = tic; % Start the total runtime stopwatch
                        tLoadStart = tic;
fprintf('\tLoad ground truth card readings\n');
load('cornerColorCardTrainingSetGroundTruth.mat')
files = table(gTruth.DataSource.Source,'VariableNames',{'imageFilename'});
TrainingCards = [files, gTruth.LabelData];
                        tLoadEnd = toc(tLoadStart)

clear tLoadStart tLoadEnd files

%% Split TrainingCards {70% Train - 10% Validation - 20% Test}

                        tSplitStart = tic;
fprintf('\tSplit TrainingCards\n');
rng(0)
shuffledIndices = randperm(height(TrainingCards));
idx = floor(0.7 * height(TrainingCards));

trainingIdx = 1:idx;
trainingDataTbl = TrainingCards(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx+1 + floor(0.1 * length(shuffledIndices));
validationDataTbl = TrainingCards(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = TrainingCards(shuffledIndices(testIdx),:);
                    tSplitEnd = toc(tSplitStart)

clear tSplitStart tSplitEnd shuffledIndices idx trainingIdx validationIdx testIdx

%% Convert datasets to imageDatastore and boxLabelDatastore

                        tConvertStart = tic;
fprintf('\tConvert datasets to imageDatastore and boxLabelDatastore\n');
imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'corner'));

imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,'corner'));

imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'corner'));
                    tConvertEnd = toc(tConvertStart)

clear tConvertStart tConvertEnd

%% Combine datastores

                        tCombineStart = tic;
fprintf('\tCombine datastores\n');
trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);
                tCombineEnd = toc(tCombineStart)

clear tCombineStart  tCombineEnd    imdsTrain bldsTrain ...
      imdsValidation bldsValidation imdsTest  bldsTest

%% Creating the network

                        tCreateNetStart = tic;
fprintf('\tCreating the network\n');
% Provide input size for the network
inputSize = [227 227 3];

% Preprocess the training data
trainingData = transform(trainingData, ...
                           @(data)preprocessData(data,inputSize));
numAnchors = 3;
anchorBoxes = estimateAnchorBoxes(trainingData,numAnchors);

% Specify the first network of three: feature extraction
featureExtractionNetwork = squeezenet;

% Specify feature extraction layer and number of classes to detect
featureLayer = 'fire6-relu_squeeze1x1'; numClasses = 1;
lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes, ...
                                    featureExtractionNetwork,featureLayer);
                tCreateNetEnd = toc(tCreateNetStart)

clear tCreateNetStart tCreateNetEnd preprocessedTrainingData

%% Train and validation data augmentation before network training

                        tAugmentStart = tic;
fprintf('\tTrain and validation data augmentation before network training\n');
validationData = transform(validationData,@(data)preprocessData(data,inputSize));
augmentedData = cell(4,1);
for k = 1:4
    data = read(trainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
end
figure; montage(augmentedData,'BorderSize',10);
                    tAugmentEnd = toc(tAugmentStart)

clear tAugmentStart tAugmentEnd augmentedTrainingData augmentedData

%% Train network

                        tTrainNetStart = tic;
fprintf('\tTrain network\n');
options = trainingOptions('sgdm',...
    'MaxEpochs',5,...
    'MiniBatchSize',2,...
    'InitialLearnRate',1e-3,...
    'CheckpointPath',tempdir,...
    'ValidationData',validationData,...
    'Plots','training-progress');
doTraining = true;
if doTraining
    [detector, info] = trainFasterRCNNObjectDetector(trainingData,lgraph,options, ...
                       'NegativeOverlapRange',[0 0.3], 'PositiveOverlapRange',[0.6 1]);
else
    load('squeezenet_unFrozen_32_10_shuffled_30_60_color_mixedCorner.mat');
    detector = pretrained.detector;
end
                    tTrainNetEnd = toc(tTrainNetStart)

clear tTrainNetStart tTrainNetEnd

%% Check am example image

                        tDisplay1Start = tic
fprintf('\tCheck an example image\n');
I = imread(testDataTbl.imageFilename{10});
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure; imshow(I);
                    tDisplay1End = toc(tDisplay1Start)

clear tDisplay1Start tDisplay1End 

%% Evaluate Detector on entire test set

                        tEvaluateTestStart = tic;
fprintf('\tEvaluate Detector on entire test set\n');
testData = transform(testData,@(data)preprocessData(data,inputSize));
detectionResults = detect(detector,testData,'MinibatchSize',4);
[ap, recall, precision] = evaluateDetectionPrecision(detectionResults,testData);
                tEvaluateTestEnd = toc(tEvaluateTestStart)

clear tEvaluateTestStart tEvaluateTestEnd


%% Display precision/recall (PR) curve

                        tDisplay2Start = tic;
fprintf('\tDisplay precision/recall (PR) curve\n');
figure; plot(recall,precision)
xlabel('Recall'); ylabel('Precision');
grid on;
title(sprintf('Average Precision = %.2f', ap));
                        tDisplay2End = toc(tDisplay2Start)
        tEnd = toc(tStart) % Stop the total runtime stopwatch

clear tDisplay2Start tDisplay2End tStart tEnd

%% Helper functions
function data = preprocessData(data,targetSize)
    % Resize image and bounding boxes to targetSize.
    sz = size(data{1},[1 2]);
    scale = targetSize(1:2)./sz;
    data{1} = imresize(data{1},targetSize(1:2));

    % Sanitize box data, if needed.
    data{2} = helperSanitizeBoxes(data{2}, sz);

    % Resize boxes.
    data{2} = bboxresize(data{2},scale);
end

%helperSanitizeBoxes Sanitize box data.
% This example helper is used to clean up invalid bounding box data. Boxes
% with values <= 0 are removed and fractional values are rounded to
% integers.
%
% If none of the boxes are valid, this function passes the data through to
% enable downstream processing to issue proper errors.

% Copyright 2020 The Mathworks, Inc.

function boxes = helperSanitizeBoxes(boxes, imageSize)
    persistent hasInvalidBoxes
    valid = all(boxes > 0, 2);
    if any(valid)
        if ~all(valid) && isempty(hasInvalidBoxes)
            % Issue one-time warning about removing invalid boxes.
            hasInvalidBoxes = true;
            warning('Removing ground truth bouding box data with values <= 0.')
        end
        boxes = boxes(valid,:);
        boxes = roundFractionalBoxes(boxes, imageSize);
    end
end

function boxes = roundFractionalBoxes(boxes, imageSize)
    % If fractional data is present, issue one-time warning and round data and
    % clip to image size.
    persistent hasIssuedWarning

    allPixelCoordinates = isequal(floor(boxes), boxes);
    if ~allPixelCoordinates

        if isempty(hasIssuedWarning)
            hasIssuedWarning = true;
            warning('Rounding ground truth bounding box data to integer values.')
        end

        boxes = round(boxes);
        boxes(:,1:2) = max(boxes(:,1:2), 1); 
        boxes(:,3:4) = min(boxes(:,3:4), imageSize([2 1]));
    end
end