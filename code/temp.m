rootDir = 'C:\Users\brownjw1\Documents\CSSE_463_Image Recognition\Final_Project\CSSE-463-Poker-Player-main\';
trainDir = [rootDir '\dataset\train\train'];
trainLabelDir = [rootDir '\dataset\train_labels_only.csv'];

trainImages = imageDatastore(...
   trainDir, ...
   'IncludeSubfolders',true, ...
   'LabelSource', 'none');

tempLabels = readtable(trainLabelDir);
trainLabels = table2cell(tempLabels);
trainLabels = strcat(trainLabels(:,1),trainLabels(:,3));

trainImages.Labels = categorical(trainLabels);

% for i = 1:size(trainLabels, 1)
%     for j = 1:size(trainLabels, 1)
%         if(contains(trainImages.Files(i), trainLabels(j, 1)))
%             extractedLabels(i) = trainLabels(j, 4);
%         end
%     end
% end

% Make datastores for the validation and testing sets similarly.

%xTrain = imageDatastoreReader(trainImages);
%yTrain = trainImages.Labels;

% Make datastores for the validation and testing sets similarly.

fprintf('Read images into datastores\n');

% Create Transfer Network
network = alexnet;
inputSize = network.Layers(1).InputSize;

transferredLayers = network.Layers(1:end-3);

[imgTrain,imgValidation] = splitEachLabel(trainImages,0.7,'randomized');

layers = [
  transferredLayers
  fullyConnectedLayer(51, 'WeightLearnRateFactor', 30, 'BiasLearnRateFactor', 20)
  softmaxLayer
  classificationLayer];

% Scale Images
% imgTrain = gray2rgb(imgTrain);
% imgValidation = gray2rgb(imgValidation);
augmentedTrainingImages = augmentedImageDatastore(inputSize(1:2),imgTrain,'ColorPreprocessing', 'gray2rgb');
augmentedValidationImages = augmentedImageDatastore(inputSize(1:2),imgValidation,'ColorPreprocessing', 'gray2rgb');
 
% augmentedTrainingImages = augmentedImageDatastore(inputSize(1:2),augmentedTrainingImages);
% augmentedValidationImages = augmentedImageDatastore(inputSize(1:2),augmentedValidationImages);

% Train Model
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');
%     'ValidationData',augmentedValidationImages, ...
%     'ValidationPatience',5, ...
%     'ValidationFrequency',5, ...


transferNetwork = trainNetwork(augmentedTrainingImages, layers, options);

% Test Model
% testImages = imageDatastore(...
%    [rootdir 'test'], ...
%    'IncludeSubfolders',true, ...
%    'LabelSource', 'foldernames');
% augmentedTestImages = augmentedImageDatastore(inputSize(1:2),testImages);
% 
% [YPred,scores] = classify(transferNetwork,augmentedTestImages);