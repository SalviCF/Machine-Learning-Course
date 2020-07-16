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
                 hidden_layer_size, (input_layer_size + 1)); % 25 x 401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % 10 x 26

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
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

inputs = X; % will be used for the gradient. The bias term will be added one at the time
% Add ones to the X data matrix
X = [ones(m, 1) X]; % X is activation 1

% Feed-forward
z2 = X * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1), a2]; % add bias to second layer
z3 = a2 * Theta2';
a3 = sigmoid(z3);
h_theta = a3; % activations of the outpout units for each example

% Compute the cost J
y_vectors = zeros(m, num_labels); % recoding the y-labels
for i = 1:m 

    row = zeros(1, num_labels); 
    index = y(i);
    row(index) = 1;
    y_vectors(i, :) = row;
    
end
% better way: y_vectors = [1:num_labels] == y

cost_matrix = - y_vectors .* log(h_theta) - (1 - y_vectors) .* log(1 - h_theta); 
J = ( (1 / m) * sum(sum(cost_matrix)) ); % sum(cost_matrix(:))

% Adding regularization term to J
Theta1_squared = Theta1(:,2:end).^2; % excluding bias term
Theta2_squared = Theta2(:,2:end).^2; % excluding bias term
regularization_term = (lambda / (2*m)) * (sum(sum(Theta1_squared)) + sum(sum(Theta2_squared)));

J = J + regularization_term;

% Backpropagation
% I have h_theta in a vector but I will do it example by example (not vectorized)
% I know I'm computing things twice...

for t = 1:m % for each example in X === inputs

    % 1.- Feed-forward 
    a_1 = [1 ; inputs(t, :)']; % input layer plus the bias term as a column vector
    z_2 = Theta1 * a_1;
    a_2 = [1 ; sigmoid(z_2)]; % sigmoid + add bias term to the second layer
    z_3 = Theta2 * a_2;
    a_3 = sigmoid(z_3);
    htheta = a_3; % column vector of the activations of the outpout units for t-th
    
    % 2.- Errors of the output layer
    y_encoded = zeros(1, num_labels)'; 
    y_encoded(y(t)) = 1; % column vector enconded of the current example
    delta_3 = htheta - y_encoded; % error of each output unit for the current example
    
    % 3.- Errors of the hidden layer
    % https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/y5QF9fbIRI6UBfX2yMSOlg
    % https://www.coursera.org/learn/machine-learning/resources/Uuxg6
    Theta2_nb = Theta2(:, 2:end); % excluding bias term
    delta_2 = (Theta2_nb' * delta_3).* sigmoidGradient(z_2);
    
    % 4.- Accumulate the gradient
    Theta1_grad = Theta1_grad + delta_2 * a_1';
    Theta2_grad = Theta2_grad + delta_3 * a_2';
    
end 

% 5.- Unregularized gradient
Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;

% Adding regularization term (excluding the first column)
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m) * Theta1(:, 2:end); 
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m) * Theta2(:, 2:end); 

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
