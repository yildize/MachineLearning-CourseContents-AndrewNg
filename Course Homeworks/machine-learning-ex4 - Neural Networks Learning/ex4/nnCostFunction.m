function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

    X = [ones(m,1),X];
    
    %y vektöründeki 10,1,2,..9 de?erlerini vektörlerle de?i?tirmeliyim
    %çünkü output 0 ile 1 aras? de?er alabilir! Cost buna göre hesaplanacak
    %o zaman diyece?im ki mesela 10 yerine [1 0 0 ....0] olsun yani ilk
    %output unit 1 olsun di?erleri 0 olsun!
    
    new_y = zeros(m,num_labels);%5000x10'luk bir matris! Her training set için her output unitin de?eri belli (0 or 1)!  
    
    for i=1:m
        if y(i)==10
                new_y(i,10) = [1]; % 0'a kar??l?k geliyor! 10. unit 1 diyoruz!
        elseif y(i)==1
                new_y(i,1) = [1]; % 1'e kar??l?k geliyor! 1. unit 1 diyoruz!
        elseif y(i)==2
                new_y(i,2) = [1]; % 2'ye kar??l?k geliyor! 2. unit 1 diyoruz!
        elseif y(i)==3
                new_y(i,3) = [1]; 
        elseif y(i)==4
                new_y(i,4) = [1]; 
        elseif y(i)==5
                new_y(i,5) = [1]; 
        elseif y(i)==6
                new_y(i,6) = [1];
        elseif y(i)==7
                new_y(i,7) = [1]; 
        elseif y(i)==8
                new_y(i,8) = [1]; 
        elseif y(i)==9
                new_y(i,9) = [1]; % 9'a kar??l?k geliyor! 9. unit 1 diyoruz!        
        end
    end
    
    y = new_y;
    
%     y(1:500,10) = [1]; % 0'a kar??l?k geliyor! 10. unit 1 diyoruz!
%     y(501:1000,1) = [1]; % 1'e kar??l?k geliyor!
%     y(1001:1500,2) = [1];
%     y(1501:2000,3) = [1];
%     y(2001:2500,4) = [1];
%     y(2501:3000,5) = [1];
%     y(3001:3500,6) = [1];
%     y(3501:4000,7) = [1];
%     y(4001:4500,8) = [1];
%     y(4501:5000,9) = [1]; %9'a kar??l?k geliyor!
    
    %FORWARD PROPAGATION ILE HER TRAINING EX ICIN HER OUTPUT UNITIN
    %URETECEGI SONUCU BULALIM:
    
    z2 = X*Theta1';
    a2 = sigmoid(z2);
    a2 = [ones(m, 1) a2]; %?lk sütuna a0=1'ler eklensin. (5000x26)

    z3 = a2*Theta2';
    a3 = sigmoid(z3); %5000x10 Her satirda, her training line için 10 adet tahmin var. Her satiri 1x10 luk bir hipotez vektörü olarak da düsünebiliriz!
    
    h = a3;

    CostVector = (-1/m)*(y.*log(h) + (1-y).*log(1-h)); % 5000x10'lik cost vector. Her sat?rda, ilgili training ex kullan?ld?, her output unit için 10 adet cost var!
    J =  sum(sum(CostVector));
    


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
    DELTA1 = zeros(size(Theta1));
    DELTA2 = zeros(size(Theta2));

    for i=1:m
        z2 = (X(i,:)*Theta1')'; %25x1 vektör. i. training ex için z2 vektörü!
        a2 = sigmoid(z2);
        a2 = [1;a2]; %26x1: [a0,a1,a2,...,a25]'
        
        z3 = a2'*Theta2'; %1x10 vektör!
        a3 = sigmoid(z3);
        
        delta3 = (a3-y(i,:))'; %i. training ex için: 10x1 lik hata vektörü.
        delta2 = ((Theta2')*delta3).*(a2.*(1-a2)); % 26x1 lik hata vektörü: Delta0,Delta1,Delta2,....Delta25. Delta0'? REMOVE edece?iz! Çünkü anlams?z!
        delta2 = delta2(2:end); %25x1
        
        DELTA1 = DELTA1 +  delta2*X(i,:);  %25x401 yani Theta1 ile ayn? boyutta!
        DELTA2 = DELTA2 +  delta3*a2';  %10x26 yani Theta2 ile ayn? boyutta!      
    end
    
    Theta1_grad = DELTA1./m;
    Theta2_grad = DELTA2./m;

    


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Once J icin regularization term hesaplayal?m:
    %First Web(Theta1) Reg Term.
    %Bias termleri dahil etmiyoruz. O da ilk columnlarda Theta0'lar var!
    
    JRegTheta1 = sum(sum((Theta1(:,2:end)).^2)); %Theta1 deki ?lk sütunlar hariç her weight'in karesini al?p toplad?k!
    JRegTheta2 = sum(sum((Theta2(:,2:end)).^2)); %Theta1 deki ?lk sütunlar hariç her weight'in karesini al?p toplad?k!
    JRegTerm = (lambda/(2*m)).*(JRegTheta1+JRegTheta2);
    J = J + JRegTerm;
    
    
%?imdi Gradients için regularization terms hesaplayal?m:
    
    GradRegTheta1 = (lambda/m).*(Theta1(:,2:end));
    GradRegTheta2 = (lambda/m).*(Theta2(:,2:end));
    

    GradRegTheta1 = [zeros(size(Theta1,1),1),GradRegTheta1]; %?lk sütuna 0 ekliyorum
    GradRegTheta2 = [zeros(size(Theta2,1),1),GradRegTheta2]; %?lk sütuna 0 ekliyorum
    
    Theta1_grad = Theta1_grad + GradRegTheta1 ;
    Theta2_grad = Theta2_grad + GradRegTheta2 ;
     
    



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
