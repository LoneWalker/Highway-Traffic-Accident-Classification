This is for the class "shoulder vs non-shoulder". 
It is going to detect whetehr or not an accident occurs at shoulder area or otherwise. 
This folder contains the code for various logistic regression models and they correspond 
in the following way: 

Logistic regression					- shoulderClassification_v1.m
Bayesian logistic regression		- shoulderClassification_v2.m
Dual logistic regression			- shoulderClassification_v3.m
Dual Bayesian logistic regression	- shoulderClassification_v4.m
Kernel logistic regression			- shoulderClassification_v5.m
Relevance vector regression			- shoulderClassification_v6.m

Within each implementation, one can choose to change the color space paramter. 
For some of the models, one can choose additional parameters such as var_prior, lambda, and nu. 

* As default, each of the model runs with a stratified 10-fold (k=10) cross validation. If one 
  wishes to change the value of k, they may do so in the data_preprocessing.m file and change
  CVO = cvpartition(w,'k',10);
  the value of 'k' to the desired value. 
* As default, each of the implementation will be run in loop for different parameter setting, and 
  the loop for each parameter is respectively labeled. 
  If one only wishes to run for one iteration for each parameter, then simpily indicate the value 
  for that parameter. 
