function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

H = X * theta;
% remove 1st row of θ
theta = theta(2:end, :);
% Note that you should not regularize the θ0 term.
J = 1 / (2 * m) * sum((H - y).^2) + lambda * 1 / (2 * m) * sum(theta.^2);

% X without 1st col
X_no_1st_col = X(:, 2:end);
grad(1) = 1 / m * sum(H - y);
grad(2:end) = (1 / m * (sum((H - y) .* X_no_1st_col, 1))' + lambda / m * theta);

% grad(2:end) = (1 / m * (sum((H - y) .* X_no_1st_col))' + lambda / m * theta); % wrong, need to use sum(A, 1), not sum(A)
% if m is only one row: 
% wrong:   sum([1, 2, 3]) is 6 --> Ι stuck here for 3 hours!
% correct: sum([1, 2, 3], 1) is still [1, 2, 3]
% =========================================================================
end