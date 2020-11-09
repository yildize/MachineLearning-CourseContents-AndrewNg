function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

%Hipotez, hi(x)= sigmoid(X(i,:)*theta) ?eklinde bulunabilir!

%VERILEN PARAMETRE DEGERLER? ?Ç?N COST FUNCTION'I HESAPLAYALIM!
for i=1:m
    J = J + ((1/m)*(-y(i)*log(sigmoid(X(i,:)*theta)) - (1-y(i))*log(1-sigmoid( X(i,:)*theta))));
end

%VER?LEN PARAMETRE DE?ERLER? ?Ç?N GRADIENTS'I HESAPLAYALIM!
for j=1:n %Çünkü bu X'e 1 eklenmi?!
    for i=1:m
        grad(j)= grad(j)+(1/m)*(sigmoid(X(i,:)*theta) - y(i))*X(i,j); % Önce theta0 için, sonra theta1 için, ve son olarak theta2 için!
    end
end



%BU FONKSIYONDA COSTFUNCTION VE GRADIENTS HESAPLANDI ÇÜNKÜ BU FONKS?YON
%FMINUNC ?Ç?N KULLANILACAK (ADVANCED OPTIMIZATION ALGORITHM)



% =============================================================

end
