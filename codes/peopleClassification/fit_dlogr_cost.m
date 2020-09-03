% Author: Stefan Stavrev 2013

% Description: Cost function for MAP dual logistic regression.
% Input: psi - Ix1 column vector that contains the coefficients for
%              the activation function,
%        X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I).
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        var_prior - scale factor for the prior spherical covariance.
% Output: L - the value of the cost function evaluated at psi,
%         g - Ix1 gradient vector,
%         H - IxI Hessian matrix containing the second derivatives.
function [L, g, H] = fit_dlogr_cost (psi, X, w, var_prior, bay_state)   
    I = size(X,2);
    D = size(X,1) - 1;
    
    % Initialize.
    if bay_state == 1
        % L = I * (-log (mvnpdf (psi, zeros(I,1), var_prior*eye(I))));
        L = I/2 * ((var_prior*I)+sum(psi.^2/var_prior));
        g = I * psi / var_prior;
        H = I * diag(repmat(1/var_prior,1,I));
    else
        L = 0; 
        g = 0; 
        H = 0; 
    end
   
    predictions = sigmoid((X*psi)' * X);
    for i = 1 : I
        % Update L.
        y = predictions(i);
        if w(i) == 1
            L = L - log(y+eps);
        else
            L = L - log(1-y+eps);
        end
        
        % Update g and H.
        temp = X' * X(:,i);
        g = g + (y-w(i)) * temp;
        H = H + y * (1-y) * (temp * temp');
    end
end