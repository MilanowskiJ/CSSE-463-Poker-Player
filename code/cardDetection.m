% Card Detector using Feature Extraction (AlexNet) and SVM's
% CSSE 463 - Image Recognition
% Brendan Boewe, Jared Brown, Jake Milinowski
% Chirag Sirigere, and William Thesken
% Code for this and all related files and functions completed July/August 2021

% This main function calls a suitDetector and rankDetector functions which
% respectively return the most likely suits and ranks for the given test
% dataset.  This determines the overall accuracy by comparing the suit and
% rank accuracy for each file.

clear all; close all; clc;

suit = suitDetector();
rank = rankDetector();

count = 0;
for i = 1:length(suit)
   suitFilename = split(cellstr(suit(i,3)), '\');
   rankFilename = split(cellstr(rank(:,3)), '\');
   
   [row ~] = find(strcmp(cellstr(suitFilename(end)),cellstr(rankFilename(:,end))));
   if contains(cellstr(suit(i,1)), cellstr(suit(i,2))) && ...
           contains(cellstr(rank(row,1)), cellstr(rank(row,2)))
       count = count + 1;
   end
end

overallAccuracy = count/length(suit)













