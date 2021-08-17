% Card Recognition Demo
% Brendan  Boewe, Jared Brown. Will Thesken, Jake Milanowski, Chirag Sirigere
% CSSE-463 Image Recognition with Dr. Boutell
% 08/17/2021

clc;
clear all;
close all;
% Loading Trained Networks
CroppedRankNetwork = load ('croppedRankNetwork.mat');
CroppedSuitNetwork = load ('croppedSuitNetwork.mat');


% Define file path for example demo images and load into datastore
demoDir = 'C:\Users\brownjw1\Desktop\DemoImages';
demoImages = imageDatastore(...
   demoDir, ...
   'IncludeSubfolders',true, ...
   'LabelSource', 'foldernames');
DemoLabels = ["AD"; "JC?"; "KC"; "4D"; "3D"; "AS"; "7C"; "KS"];


% Augmentimages to resize for input into CNNs
network = alexnet;
inputSize = network.Layers(1).InputSize;
augmentedDemoImages = augmentedImageDatastore(inputSize(1:2),demoImages,'ColorPreprocessing', 'gray2rgb');


% Classify Images using suit and rank CNNs
[YPredRanks,scoresRanks] = classify(CroppedRankNetwork.transferNetwork,augmentedDemoImages);
[YPredSuits,scoresSuits] = classify(CroppedSuitNetwork.transferNetwork,augmentedDemoImages);

% Concatenating Results
for k= 1:length(YPredSuits)
    FinalPredictedLabels(k,1) = strcat(string(YPredRanks(k)), string(YPredSuits(k)));
end
zComparison = [FinalPredictedLabels DemoLabels];


% Creating Figure
for k=1:length(demoImages.Files)
    image = imread(string(cellstr(demoImages.Files(k))));
    figure(1);
    subplot(2,4,k)
    hold on;
    title('P: ' + zComparison(k,1) + '   A: ' + zComparison(k,2));
    imshow(image);
end



























