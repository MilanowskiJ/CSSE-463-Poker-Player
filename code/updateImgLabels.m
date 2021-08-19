function [updatedLabels] = updateImgLabels(imageDatastore,label)
%updateImgLabels

% Takes in a label and image datastore.  For all images in the datastore
% without the passed in label, relabels them to be 'N'+label. Returns a
% list of updated labels.

newLabels = imageDatastore.Labels;
for i = 1:length(newLabels)
   if contains(cellstr(imageDatastore.Labels(i)), label)
       newLabels(i) = cellstr(label);
   else
       newLabels(i) = cellstr(strcat('N',label));
   end   
end
updatedLabels = categorical(newLabels);

end

