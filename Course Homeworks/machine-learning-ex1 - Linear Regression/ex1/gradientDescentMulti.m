function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

SlopeTheta1=0;
SlopeTheta2=0;
SlopeTheta3=0;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


       for i=1:m
        hi = (theta')*X(i,:)'; %Hypothesis result for every training ex.
        SlopeTheta1=SlopeTheta1+alpha*(1/m)*(hi-y(i));
        SlopeTheta2=SlopeTheta2+alpha*(1/m)*((hi-y(i))*X(i,2));%Error for every training ex.  
        SlopeTheta3=SlopeTheta3+alpha*(1/m)*((hi-y(i))*X(i,3));
       end
       theta(1)= theta(1)- SlopeTheta1;
       theta(2)= theta(2)- SlopeTheta2;
       theta(3)= theta(3)- SlopeTheta3;
       
       SlopeTheta1=0;
       SlopeTheta2=0;
       SlopeTheta3=0;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end