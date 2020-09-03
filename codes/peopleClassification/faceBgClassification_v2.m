% San Yeung, Missouri S&T 
% 4-18-2017
% CS 6001 Assignment 6 - Segment classification using logistic regression. 

clc;
clear all;
close all;
disp('Bayesian logistic regression model');

% Data preprocessing.
if exist('X','var')==0
    data_preprocessing;
end

loop = 3;
avg_miss_detection = 0; 
avg_false_alarm = 0;
for jj = 1:loop
    %% Learning phase: to learn the parameters to the sigmoid function by fitting
    %  a logistic regression model.
    var_prior = 10;
    %var_prior = [];
    D = size(X,1);
    initial_phi = rand(D,1);
    [predictions,phi] = fit_blogr (X, w, var_prior, X_test, initial_phi, 1);

    %% Evaluation phase.
    % Miss detection.
    miss_detection = 0; 
    for ii = size(allBgTestIms,2)+1:size(w_gt,1)
        if(predictions(ii) >= 0.5)
            % predicted_class = 1;
        else
            % predicted_class = 0;
            miss_detection = miss_detection+1;
        end
    end
    miss_detection = miss_detection/(ii-size(allBgTestIms,2));
    avg_miss_detection = avg_miss_detection + miss_detection;

    % False alarms. 
    false_alarm = 0;
    for ii = 1:size(allBgTestIms,2)
        if(predictions(ii) >= 0.5)
            % predicted_class = 1;
            false_alarm = false_alarm+1;
        else
            % predicted_class = 0;
        end
    end
    false_alarm = false_alarm/ii;
    avg_false_alarm = avg_false_alarm + false_alarm;
end
disp(['miss_detection: ',num2str(avg_miss_detection/loop)]);
disp(['false_alarm: ',num2str(avg_false_alarm/loop)]);
disp('End of program');