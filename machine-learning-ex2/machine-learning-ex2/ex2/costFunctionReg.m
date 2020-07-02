function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h_theta = sigmoid(X * theta);

cost_vector = - y .* log(h_theta) - (1 - y) .* log(1 - h_theta);  

regularization_term = theta(2:size(theta)).^2; % excluding theta_0

J = ( (1 / m) * sum(cost_vector) ) + ( (lambda / (2*m)) * sum(regularization_term) ); 

partial_j_0 = (1 / m) * sum( (h_theta - y) .*  X(:, 1));

other_partials = ( (1/m) * sum((h_theta - y) .* X(:, 2:size(X,2))) ) + (lambda/m) * theta(2:size(theta))';

grad = [partial_j_0, other_partials];

% =============================================================

end
