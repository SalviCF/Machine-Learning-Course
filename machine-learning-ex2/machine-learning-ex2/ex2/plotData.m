function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% Find Indices of Positive and Negative Examples
pos_indices = find(y == 1); % help find 
neg_indices = find(y == 0);

% Plot Examples
% https://www.mathworks.com/help/matlab/ref/plot.html

% x-axis: X(pos_indices, 1): positive indices from y in column 1 of X
% y-axis: X(pos_indices, 2): positive indices from y in column 2 of X
% 'k+' means black cross, width 2 for the lines of the cross and size 7
plot(X(pos_indices, 1), X(pos_indices, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);

% x-axis: X(neg_indices, 1): negative indices from y in column 1 of X
% y-axis: X(neg_indices, 2): negative indices from y in column 2 of X
% 'ko' means black circumference
% 'y' means yellow fill for the circle of size 7
plot(X(neg_indices, 1), X(neg_indices, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

% =========================================================================

hold off;

end
