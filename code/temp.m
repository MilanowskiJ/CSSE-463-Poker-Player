rootDir = 'C:\Users\milanoj\Documents\Classes\CSSE463\CSSE-463-Poker-Player\';
trainDir = [rootDir '\dataset\train\train'];
trainLabelDir = [rootDir '\dataset\train_labels_only.csv'];

trainImages = imageDatastore(...
   trainDir, ...
   'IncludeSubfolders',true, ...
   'LabelSource', 'none');

tempLabels = readtable(trainLabelDir);
trainLabels = table2cell(tempLabels);
trainLabels = strcat(trainLabels(:,1),trainLabels(:,3));

trainImages.Labels = trainLabels;

% for i = 1:size(trainLabels, 1)
%     for j = 1:size(trainLabels, 1)
%         if(contains(trainImages.Files(i), trainLabels(j, 1)))
%             extractedLabels(i) = trainLabels(j, 4);
%         end
%     end
% end

% Make datastores for the validation and testing sets similarly.

fprintf('Read images into datastores\n');

%trainImages.Files == 

%xTrain = imageDatastoreReader(trainImages);
%yTrain = trainImages.Labels;