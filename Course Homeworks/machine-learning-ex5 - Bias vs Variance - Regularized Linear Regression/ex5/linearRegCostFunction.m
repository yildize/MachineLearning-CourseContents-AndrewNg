function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta)-1;

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%Cost without regularization terms
hi = X*theta;
singleLineCostVector = (hi-y).^2;
J = (1/(2*m))*sum(singleLineCostVector);

%Adding regularization terms
regTerm = theta;
regTerm(1) = [];
regTerm = regTerm.^2;
TotalRegTerm = (lambda/(2*m))*sum(regTerm);
J = J + TotalRegTerm ;


%Gradients w/o regularization terms
grad = (1/m)*((X')*(hi-y));

%Add gradient terms for j>0
theta(1) = 0;
GradRegTerm = (lambda/m)*theta;
grad = grad + GradRegTerm;

% =========================================================================

grad = grad(:);

end
