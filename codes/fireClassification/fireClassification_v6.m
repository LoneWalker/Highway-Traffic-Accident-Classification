% San Yeung, Missouri S&T 
% 5-8-2017
% CS 6001 semester project

clc;
close all;
disp('Relevance vector logistic regression model');

% Open file
fileID = fopen('rvlr.txt','w');
colorSpace = cell(6,1);
colorSpace{1} = 'RGB';
colorSpace{2} = 'Red';
colorSpace{3} = 'Gray';
colorSpace{4} = 'HSV';
colorSpace{5} = 'YCbCr';
colorSpace{6} = 'Gradient';

%% nu loop
for nuVal = [.001,.01,.1,1,10,100]
    disp(['nuVal: ',num2str(nuVal)]);
    fprintf(fileID,'\n%10s %f\n','nuVal:',nuVal);
%% lambda loop
for lambdaVal = [.001,.01,.1,1,10,100]
    disp(['lambdaVal: ',num2str(lambdaVal)]);
    fprintf(fileID,'\n\n%10s %f\n','lambdaVal:',lambdaVal);
for j = 1:6
% for colorSpace = [string('RGB'),string('Red'),string('Gray'),...
%         string('HSV'),string('YCbCr'),string('Gradient')]
    % Data preprocessing
    imScale = .25;
    [allRegTrainIms,allIncidentTrainIms,X,w,CVO]...
        = data_preprocessing(colorSpace{j},imScale);
    fprintf(fileID,'%s\n',colorSpace{j});
    
    avg_miss_detection = 0; 
    avg_false_alarm = 0;

    for i = 1:CVO.NumTestSets
        disp(['Iteration ', num2str(i)]);
        %% Learning phase: to learn the parameters to the sigmoid function by fitting
        %  a logistic regression model.
        D = CVO.TrainSize(i);
        initial_psi = rand(D,1);
        kernel = @kernel_gauss;
        lambda = lambdaVal;
        nu = nuVal;
        X_k = X(:,CVO.training(i));
        w_k = w(CVO.training(i),:);
        X_test_k = X(:,CVO.test(i));
        [predictions,psi] = fit_rvc (X_k, w_k, nu, X_test_k,...
            initial_psi, kernel, lambda);

        %% Evaluation phase.
        miss_detection = 0;
        false_alarm = 0;
        w_gt_k = w(CVO.test(i),:);
        for ii = 1:size(predictions,2)
            if (predictions(ii) >= 0.5) && (w_gt_k(ii) == 0)
                % predicted class = 1
                false_alarm = false_alarm+1;
            elseif (predictions(ii) < 0.5) && (w_gt_k(ii) == 1)
                % predicted class = 0
                miss_detection = miss_detection+1;
            end
        end
        miss_detection = miss_detection/size(predictions,2);
        avg_miss_detection = avg_miss_detection + miss_detection;
        false_alarm = false_alarm/size(predictions,2);
        avg_false_alarm = avg_false_alarm + false_alarm;
    end

    disp(['miss_detection: ',num2str(avg_miss_detection/CVO.NumTestSets)]);
    disp(['false_alarm: ',num2str(avg_false_alarm/CVO.NumTestSets)]);
    fprintf(fileID,'%f\n',avg_miss_detection/CVO.NumTestSets);
    fprintf(fileID,'%f\n',avg_false_alarm/CVO.NumTestSets);
end
end
end
% Close file
fclose(fileID);
disp('End of program');

