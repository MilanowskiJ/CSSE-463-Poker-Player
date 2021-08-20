clc;
clear all;
rootDir = 'C:\Users\brownjw1\Documents\CSSE_463_Image Recognition\Final_Project\CSSE-463-Poker-Player\';
trainDir = [rootDir '\dataset\croppedInputData\suit'];

trainImages = imageDatastore(...
   trainDir, ...
   'IncludeSubfolders',true, ...
   'LabelSource', 'foldernames');



% Creating Random Test Set
 p = randperm(length(trainImages.Files), 200);
 
 TestImages = trainImages.Files(p); 
 
 imgTest = imageDatastore(...
   TestImages, ...
   'IncludeSubfolders',true, ...
   'LabelSource', 'foldernames');
suitLabels = [imgTest.Labels(:) imgTest.Files(:)];



% Removing test images from training set
x1 = [];
for x=1:length(trainImages.Files)
    if ~(ismember(x,p(:)))
        x1 = [x1 x];
    end
    
end
 
images = trainImages.Files(x1);
trainImages = imageDatastore(...
   images, ...
   'IncludeSubfolders',true, ...
   'LabelSource', 'foldernames');

 
[imgTrain,imgValidation] = splitEachLabel(trainImages,0.7,'randomized');

[SuitPredictions, SuitTestLabels] = suit_Predictor_crop(imgTrain, imgValidation, imgTest);



% Loading data for rank detector
trainDir = [rootDir '\dataset\croppedInputData\rank'];
trainImages = imageDatastore(...
   trainDir, ...
   'IncludeSubfolders',true, ...
   'LabelSource', 'foldernames');

testfilenames = split(TestImages, '\');
allfilenames = split(trainImages.Files(:), '\');



ind = [];
for i = 1:length(allfilenames)
    file = split(trainImages.Files(i), '\');
   if ismember(cellstr(file(end)),testfilenames(:,end)) 
       ind = [ind i];
   end
end

TestImages = trainImages.Files(ind);

imgTest = imageDatastore(...
   TestImages, ...
   'IncludeSubfolders',true, ...
   'LabelSource', 'foldernames');

x1 = [];
for x=1:length(trainImages.Files)
    if ~(ismember(x,ind(:)))
        x1 = [x1 x];
    end
    
end
 
images = trainImages.Files(x1);
trainImages = imageDatastore(...
   images, ...
   'IncludeSubfolders',true, ...
   'LabelSource', 'foldernames');

 
[imgTrain,imgValidation] = splitEachLabel(trainImages,0.7,'randomized');

[RankPredictions, RankTestLabels] = rank_Predictor_crop(imgTrain, imgValidation, imgTest);


ind = [];
for i = 1:length(TestImages)
    file = split(TestImages(i), '\');
   if ismember(cellstr((file(end))),testfilenames(:,end)) 
       ind = [ind i];
   end
end

SuitPredictionsFixed = SuitPredictions(ind);
Predictions = horzcat(RankPredictions,SuitPredictionsFixed);


for c= 1:length(Predictions)
    FinalPredictions(c,1) = strcat(string(Predictions(c,1)), string(Predictions(c,2)));
end


ind = [];
for i = 1:length(imgTest.Files)
    file = split(imgTest.Files(i), '\');
   if ismember(cellstr((file(end))),testfilenames(:,end)) 
       ind = [ind i];
   end
end

suitLabels = suitLabels(ind,:);

rankFileName = split(cellstr(imgTest.Files(:)),'\');
for i = 1:length(suitLabels)
    suitFileName = split(cellstr(suitLabels(i,2)), '\');
    
    [row ~] = find(strcmp(cellstr(suitFileName(end)), cellstr(rankFileName(:,end))));
    newsuitTestLabels(i) = suitLabels(row, 1);
    
end
suitLabels = newsuitTestLabels;

for k= 1:length(RankTestLabels)
    FinalTestLabels(k,1) = strcat(string(RankTestLabels(k)), string(SuitTestLabels(k)));
end


matches2 = find(FinalPredictions == FinalTestLabels);
matches2 = FinalPredictions(matches2);
accuracy = length(matches2)/length(FinalPredictions);

fprintf('Total Model Accuracy: %0.2f\n' , accuracy);








