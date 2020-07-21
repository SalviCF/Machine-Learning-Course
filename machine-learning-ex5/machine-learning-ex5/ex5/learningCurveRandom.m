function [error_train, error_val] = ...
    learningCurveRandom(X, y, Xval, yval, lambda, rep)

% https://www.coursera.org/learn/machine-learning/discussions/weeks/6/threads/P3Cp9j_ZEeaDRA5SxbW7qQ

m = size(X, 1); % number of training examples
r = size(Xval, 1); % number of validation examples

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for i = 1:m
    % zero-out
    Jtrain = zeros(1, rep);
    Jcv = zeros(1, rep);
    for j = 1:rep
        % i random examples from training set
        indices_X = randperm(m, i); % i random indices in range 1-m
        examples_X = X(indices_X, :); % selects i random examples from X
        examples_y = y(indices_X);
        
        %  Learn parameters theta
        theta = trainLinearReg(examples_X, examples_y, lambda);
        
        Jtrain(j) = linearRegCostFunction(examples_X, examples_y, theta, 0);

        % i random examples from validation set
        indices_Xval = randperm(r, i); % i random indices in range 1-r      
        examples_Xval = Xval(indices_Xval, :); % selects i random examples from Xval
        examples_yval = yval(indices_Xval);
                 
        Jcv(j) = linearRegCostFunction(examples_Xval, examples_yval, theta, 0);
    end
          
    error_train(i) = mean(Jtrain);
    error_val(i) = mean(Jcv);
end

% -------------------------------------------------------------

% =========================================================================

end
