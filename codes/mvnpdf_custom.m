function y = mvnpdf(X, Mu, Sigma)
%MVNPDF Multivariate normal probability density function (pdf).
%   Y = MVNPDF(X) returns the probability density of the multivariate normal
%   distribution with zero mean and identity covariance matrix, evaluated at
%   each row of X.  Rows of the N-by-D matrix X correspond to observations or
%   points, and columns correspond to variables or coordinates.  Y is an
%   N-by-1 vector.
%
%   Y = MVNPDF(X,MU) returns the density of the multivariate normal
%   distribution with mean MU and identity covariance matrix, evaluated
%   at each row of X.  MU is a 1-by-D vector, or an N-by-D matrix, in which
%   case the density is evaluated for each row of X with the corresponding
%   row of MU.  MU can also be a scalar value, which MVNPDF replicates to
%   match the size of X.
%
%   Y = MVNPDF(X,MU,SIGMA) returns the density of the multivariate normal
%   distribution with mean MU and covariance SIGMA, evaluated at each row
%   of X.  SIGMA is a D-by-D matrix, or an D-by-D-by-N array, in which case
%   the density is evaluated for each row of X with the corresponding page
%   of SIGMA, i.e., MVNPDF computes Y(I) using X(I,:) and SIGMA(:,:,I).
%   If the covariance matrix is diagonal, containing variances along the 
%   diagonal and zero covariances off the diagonal, SIGMA may also be
%   specified as a 1-by-D matrix or a 1-by-D-by-N array, containing 
%   just the diagonal. Pass in the empty matrix for MU to use its default 
%   value when you want to only specify SIGMA.
%
%   If X is a 1-by-D vector, MVNPDF replicates it to match the leading
%   dimension of MU or the trailing dimension of SIGMA.
%
%   Example:
%
%      mu = [1 -1]; Sigma = [.9 .4; .4 .3];
%      [X1,X2] = meshgrid(linspace(-1,3,25)', linspace(-3,1,25)');
%      X = [X1(:) X2(:)];
%      p = mvnpdf(X, mu, Sigma);
%      surf(X1,X2,reshape(p,25,25));
%
%   See also MVTPDF, MVNCDF, MVNRND, NORMPDF.

if nargin<1
    error(message('stats:mvnpdf:TooFewInputs'));
elseif ndims(X)~=2
    error(message('stats:mvnpdf:InvalidData'));
end

% Get size of data.  Column vectors provisionally interpreted as multiple scalar data.
[n,d] = size(X);
if d<1
    error(message('stats:mvnpdf:TooFewDimensions'));
end

square_diag_x= X.^2;
y = .5* sum(diag(Sigma))+.5* sum(square_diag_x./diag(Sigma));

