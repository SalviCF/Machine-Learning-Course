function [theta, J_history, theta_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_history = zeros(num_iters, 2);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

h_theta = X * theta;
error = (h_theta - y); 

theta(1) = theta(1) - alpha * (1/m) * sum( error .* X(:,1) );
theta(2) = theta(2) - alpha * (1/m) * sum( error .* X(:,2) );

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    % Save theta vector in every iteration
    theta_history(iter, :) = theta;
       
    % Remove comment to see the evolution of the regression line as the gradient descent occurs
%     plot(X(:,2), y, 'rx', 'Markersize', 10); hold on;
%     plot(X(:,2), X*theta, '-'); hold on;
%     hold off;
%     pause(1e-07);
    
end
end
