function [accuracy] = getTestAccuracy(predictedLabels, actualLabels)
%getTestAccuracy

% Takes the predicted labels and compares them to the actualLabels to
% determine an overall accuracy, which is returned.

count = 0;
for i = 1:length(predictedLabels)
    if strcmp(cellstr(predictedLabels(i)), cellstr(actualLabels(i)))
        count = count + 1;
    end
end

accuracy = count/length(predictedLabels);

end

