clc; clear all; close all;

%trainingImageAccidentDirectory = 'testingImages\accident\';
%trainingImageNonAccidentDirectory = 'testingImages\non_accident\';
%trainingImageNonAccidentDirectory = 'testingImages\accident\Fire_not_present\';
trainingImageAccidentDirectory = 'testingImages\accident\';
trainingImageNonAccidentDirectory = 'testingImages\accident\Vehicles_not_overturned\';

testingImageAccidentDirectory = trainingImageAccidentDirectory;
testingImageNonAccidentDirectory = trainingImageNonAccidentDirectory;


colorSpace= 'RGB'; % possible colorspaces: RGB, Gray, HSV, YCbCr, HSVYCbCr, Gradient

mode_logistic_regression=1;
mode_bayesian_logistic_regression=2;
mode_dual_logistic_regression=3;
mode_dual_bayesian_logistic_regression=4;
mode_kernel_logistic_regression=5;
mode_relevance_logistic_regression=6;

iteration=10;
mode=mode_kernel_logistic_regression;

%tunign parameters
var_prior=1;

nu=5;
lambda= 1000;


% ***********************code for accident ********************************

allAccidentIms=[];
folderList = dir(trainingImageAccidentDirectory);

%[allIms,nrows,ncols,np] = getAllIms([trainingImageAccidentDirectory folderList(7).name '\'],colorSpace); % np is how many dimention in the colorspace

%for ii= 3:size(folderList,1)
for ii= 16:16%size(folderList,1)
    [allIms,nrows,ncols,np] = getAllIms([trainingImageAccidentDirectory folderList(ii).name '\'],colorSpace); % np is how many dimention in the colorspace
    if isempty(allIms), continue; end
    allAccidentIms = [allAccidentIms; allIms];    
end
totalAccidentImsTraining=size(allAccidentIms,1);


% ******************* non accident ***********************

[allNonAccidentIms,nrows,ncols,np] = getAllIms(trainingImageNonAccidentDirectory,colorSpace); % np is how many dimention in the colorspace

totalNonAccidentImsTraining=size(allNonAccidentIms,1);
total_training_images = totalAccidentImsTraining + totalNonAccidentImsTraining;

w=[ones(totalAccidentImsTraining,1); zeros(totalNonAccidentImsTraining,1)]; % assuming non accident(w=0) and accident (w=1)
X= [allAccidentIms' allNonAccidentIms'];
X=[ones(1,total_training_images);X];


%***********************code for cross validation*****************************
CVO = cvpartition(w,'k',iteration);


%***************** Testing dataset********************

%for accident image dataset

% allAccidentIms=[];
% folderList = dir(testingImageAccidentDirectory);
% 
% for ii= 3:size(folderList)
% %for ii= 5:5
%     [allIms,nrows,ncols,np] = getAllIms([testingImageAccidentDirectory folderList(ii).name '\'],colorSpace); % np is how many dimention in the colorspace
%     if isempty(allIms), continue; end
%     allAccidentIms = [allAccidentIms; allIms];    
% end
% totalAccidentImsTesting=size(allAccidentIms,1);
% 
% %for non accident image dataset
% 
% [allNonAccidentIms,nrows,ncols,np] = getAllIms(testingImageNonAccidentDirectory,colorSpace); % np is how many dimention in the colorspace
% 
% totalNonAccidentImsTesting=size(allNonAccidentIms,1);
% total_testing_images = totalAccidentImsTesting + totalNonAccidentImsTesting;
% 
% X_test=[allAccidentIms' allNonAccidentIms'];
% X_test=[ones(1,total_testing_images);X_test];


miss_detection=0;
false_alarm=0;
avg_miss_detection=0;
avg_false_alarm=0;

for j =1:CVO.NumTestSets
    %calling the appropriate logistic regression method
    
    D = CVO.TrainSize(j);
    initial_psi = rand(D,1);
    X_k = X(:,CVO.training(j));
    w_k = w(CVO.training(j),:);
    X_test_k = X(:,CVO.test(j));
    
    
    if mode==mode_logistic_regression
        D_plus_one=size(X_test_k,1);
        initial_phi= rand(D_plus_one,1);
        initial_phi=initial_phi+eps;
        [predictions,  phi] = fit_logr(X_k,w_k,var_prior,X_test_k,initial_phi);
    elseif mode == mode_bayesian_logistic_regression
        D_plus_one=size(X_test_k,1);
        initial_phi= rand(D_plus_one,1);
        initial_phi=initial_phi+eps;
        [predictions,  phi] = fit_blogr(X_k,w_k,var_prior,X_test_k,initial_phi);
    elseif mode == mode_dual_logistic_regression
        initial_psi = rand(D,1);
        initial_psi =  initial_psi+eps;
        [predictions,  phi] = fit_dlogr(X_k,w_k,var_prior,X_test_k,initial_psi);
    elseif mode == mode_dual_bayesian_logistic_regression
        initial_psi = rand(D,1);
        initial_psi =  initial_psi+eps;
        [predictions,  phi] = fit_dblogr(X_k,w_k,var_prior,X_test_k,initial_psi);
    elseif mode == mode_kernel_logistic_regression
        initial_psi = rand(D,1);
        initial_psi =  initial_psi+eps;
        kernel=@kernel_gauss;
        [predictions,  phi] = fit_klogr(X_k,w_k,var_prior,X_test_k,initial_psi, kernel , lambda);
    else %Relevance vector logistic regression
        initial_psi = rand(D,1);
        initial_psi =  initial_psi+eps;
        kernel=@kernel_gauss;
        [predictions,  phi] = fit_rvc(X_k, w_k, nu, X_test_k,initial_psi, kernel , lambda);
    end

    
    %% Evaluation phase.
    miss_detection = 0;
    false_alarm = 0;
    w_gt_k = w(CVO.test(j));
    for ii = 1:size(predictions,2)
        if (predictions(ii) < 0.5) && (w_gt_k(ii) == 1)
            % predicted class = 1
            miss_detection = miss_detection+1;
            
        elseif (predictions(ii) >= 0.5) && (w_gt_k(ii) == 0)
            % predicted class = 0
            false_alarm = false_alarm+1;
        end
    end
    miss_detection = miss_detection/size(predictions,2);
    avg_miss_detection = avg_miss_detection + miss_detection;
    false_alarm = false_alarm/size(predictions,2);
    avg_false_alarm = avg_false_alarm + false_alarm;


    % Calculating evaluation Metrics

%     miss_detection=0;
%     false_alarm=0;
%     for i = 1:total_testing_images
%         if i <= totalAccidentImsTraining % these are face images
%             if predictions(i)<.5
%                 %miss_detection= miss_detection +  abs(predictions(i)-1);
%                 miss_detection= miss_detection + 1;
%             end
%         else % these are background images
% 
%             if predictions(i)>=.5
%                 %false_alarm= false_alarm + abs(predictions(i));
%                 false_alarm= false_alarm + 1;
%             end
%         end
% 
%     end
% 
%     miss_detection = miss_detection/totalAccidentImsTraining;
%     false_alarm=false_alarm/totalNonAccidentImsTraining;
%     
%     disp(['Miss Detection=' num2str(miss_detection)]);
%     disp(['False Alarm=' num2str(false_alarm)]);
%     
%     avg_miss_detection = avg_miss_detection + miss_detection;
%     avg_false_alarm = avg_false_alarm + false_alarm;
    
end

avg_miss_detection = avg_miss_detection/iteration;
avg_false_alarm =  avg_false_alarm/iteration;

disp(['Average Miss Detection=' num2str(avg_miss_detection)]);
disp(['Average False Alarm=' num2str(avg_false_alarm)]);
