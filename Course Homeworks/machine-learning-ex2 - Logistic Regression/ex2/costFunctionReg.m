function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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



%Hipotez, hi(x)= sigmoid(X(i,:)*theta) Seklinde bulunabilir!

%VERILEN PARAMETRE DEGERLERI ICIN COST FUNCTION'I HESAPLAYALIM:
for i=1:m
    J = J + ((1/m)*(-y(i)*log(sigmoid(X(i,:)*theta)) - (1-y(i))*log(1-sigmoid( X(i,:)*theta))));
end

%REGULARIZATION TERM EKLENSIN:
for j=2:n
    J = J + ((lambda/(2*m))*theta(j)^2); % theta0 regularization term içinde yer almaz. theta1 to thetan 
end




%VERILEN PARAMETRE DEGERLERI ICIN GRADIENTS'I HESAPLAYALIM:
for j=1:n %Çünkü bu X'e 1 eklenmis
    for i=1:m
        grad(j)= grad(j)+(1/m)*(sigmoid(X(i,:)*theta) - y(i))*X(i,j); %Theta0 to Thetan
    end
end

for j=2:n %Theta1 to Thetan (Theta0 hariç) regularization term eklensin:
    grad(j) = grad(j) + (lambda/m)*theta(j);
end




% =============================================================

end
