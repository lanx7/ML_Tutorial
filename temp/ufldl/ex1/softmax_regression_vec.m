function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  % X   n by examples
  % y   examples by 1, element is label 1 to 10
  % theta n by (num_classes-1)
  % V   num_classes by examples
  % V_colsum  1 by examples
  % h   num_classes by examples
  % h_ind 1 by examples
  % g   n by num_classes


  % theta=[theta, zeros(n,1)] 
  y_onehot = full(sparse(y, 1:m,1,num_classes,m));
  p_each = [exp(theta' * X);  ones(1,m)];
  % p_colsum = sum(exp(theta' * X));

  p_colsum= sum(p_each,1);
  for i = 1:size(p_each,2)
    p_each(:,i) = p_each(:,i) / p_colsum(1,i);
  end 
  h = p_each;

  % h= bsxfun (@rdivide, p_each, p_colsum); % devide V by it's columnsum
 
  
  %size(theta' * X)
  %size(sum(exp(theta' * X)))
  %size(y_onehot)
  %size(h)
  %size(X') 

  f = -sum(sum(y_onehot .* log(h)));

  h = h(1:(num_classes-1),:);
  y_onehot = y_onehot(1:(num_classes-1),:);
 
  g =  -X * (y_onehot - h)' ; 

  g=g(:); % make gradient a vector for minFunc

