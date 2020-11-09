function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

%Z B?R VEKT�R VEYA MATR?S GELEN HER Z ELEMANI ?�?N SIGMO?D HESAPLANACAK:
g = (exp(-z)+1).^-1;



% =============================================================

end
