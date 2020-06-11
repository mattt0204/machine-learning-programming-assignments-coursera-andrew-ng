function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values

m = length(y); % number of training examples
J_history = zeros(num_iters, 1); % values of cost function for iterations

for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    D=zeros(2,1);
    for i = 1:m
       D(1) = D(1)+(theta(1)+theta(2)*X(i,2)-y(i));
       D(2) = D(2)+(theta(1)+theta(2)*X(i,2)-y(i))*X(i,2);
    end
    
    D(1,1) = D(1,1)*alpha/m;
    D(2,1) = D(2,1)*alpha/m;
    theta(1) = theta(1) - D(1,1); 
    theta(2) = theta(2) - D(2,1);
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
