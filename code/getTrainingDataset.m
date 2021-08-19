function [files, labels] = getTrainingDataset(imageDatastore, label, allLabels)
%getTrainingDataset
% This function takes an imageDatastore, a selected label, and the list of
% all labels in the imageDatastore and returns a list of files and labels
% of an image training set consisting of all images of the label + the same
% amount of images divided equally between all other labels.

labelIdx = find(contains(cellstr(imageDatastore.Labels(:)), label));
totalLabel = length(labelIdx);

files = cell(1,totalLabel * 2);
labels = cell(1,totalLabel * 2);
for i = 1:totalLabel
   files(i) = cellstr(imageDatastore.Files(labelIdx(i)));
   labels(i) = cellstr(imageDatastore.Labels(labelIdx(i)));
end

allLabels = cellstr(allLabels);
othLabels = find(~contains(allLabels, label));

i = i + 1;
for k = 1:length(othLabels)
    randImgIdx = find(contains(cellstr(imageDatastore.Labels(:)),allLabels(othLabels(k))));
    randValues = randperm(length(randImgIdx), round(totalLabel/(length(allLabels) - 1)));
    for j = 1:length(randValues)
        files(i) = cellstr(imageDatastore.Files(randImgIdx(randValues(j))));
        labels(i) = cellstr(imageDatastore.Labels(randImgIdx(randValues(j))));
        i = i + 1;
    end
end

index = cellfun(@isempty, files) == 0;
files = files(index);

index = cellfun(@isempty, labels) == 0;
labels = labels(index);
end

