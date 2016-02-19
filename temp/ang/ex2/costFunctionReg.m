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
% theta_reg = zeros(size(theta))
theta_reg = theta; 
theta_reg(1) = 0;

hypothesis = sigmoid(X * theta); 
y1_term = -y .* log(hypothesis);
y0_term = (1 - y) .* log(1-hypothesis);
reg_term = lambda / (2 * m) * sum(theta_reg .^ 2)
J = sum(y1_term - y0_term) / m + reg_term; 

reg_grad = (lambda / m) .* theta_reg 
grad = (X' * (hypothesis - y) / m) + reg_grad; 


% =============================================================

end
