function [allRegTrainIms,allIncidentTrainIms,X,w,CVO]...
    = data_preprocessing(colorSpace, imScale)
% Training and testing data preprocessing.

%% File Directory
regImageDirectory = 'trainingImages\non_shoulder\';
incidentImageDirectory = 'trainingImages\shoulder\';

disp(['Color space: ',colorSpace]);

%% Prepare training data. 
[allRegTrainIms,~,~,~] = getAllIms(regImageDirectory,colorSpace, imScale); % Class background.
[allIncidentTrainIms,~,~,~] = getAllIms(incidentImageDirectory,colorSpace, imScale); % Class face. 
allRegTrainIms = allRegTrainIms';   % Each column is an image. 
allIncidentTrainIms = allIncidentTrainIms';
X = [allRegTrainIms, allIncidentTrainIms]; % Attach 1 to the start of the data vector. 
X = [ones(1,size(X,2));X];
w = [zeros(size(allRegTrainIms,2),1);ones(size(allIncidentTrainIms,2),1)];
% Create a cross-validation object
CVO = cvpartition(w,'k',10);
