function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
[m, n] = size(X); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%



%Verilen Thetas için J without regularization terms:
hi = sigmoid(X * theta); %5000x1'lik bir vector olacak. Her sat?rda 1 adet hipotez sonucu tutacak. Her training row için 1 sonuç!
CostVector = (-1/m)*(y.*log(hi) + (1-y).*log(1-hi)); % 5000x1'lik cost vector. Her sat?r?nda ilgili training row'a kar??l?k dü?en cost var!.
J =  sum(CostVector);

%Regularization terms:
RegTermsVector =(lambda/(2*m))*(theta.^2); % Her sat?rda, Theta0 dahil regularization terms tutuyor.
RegTermsVector(1) = []; %Theta0 dahil olmamas? gerekti?i için ç?kartt?k.
J = J + sum(RegTermsVector); 


grad = X'*(1/m)*(hi-y); %Her sat?r?nda Grad tutuyor. S?rayla Theta0'a göre GradJ, Theta1'e göre GradJ, ...


%Regularization terms:
RegTermsVectorForGrad =(lambda/(m))*(theta); % Her sat?rda, Theta0 dahil regularization terms tutuyor.
RegTermsVectorForGrad(1) = 0; %Theta0 a göre GradJ'e reg term eklenmedi?i için bunu 0 yapt?m!
grad = grad +  RegTermsVectorForGrad;



% =============================================================


end
