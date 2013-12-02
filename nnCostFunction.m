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
warning("off");
X = [ones(m, 1) X]; % adding the bias unit for all training set examples.
Z2 = Theta1 * X'; % Z2 is a matrix of 25x5000. Each vertical column is z2 for a single training example.  (Each horizontal row is z2 of an activation unit across the training set)
A2 = sigmoid(Z2); % A2 is a matrix of 25x5000. Each vertical column is the 25 activation unit in hidden layer 1 for a single training exaple.
A2 = A2'; % now this is the input into layer 3, 5000x25. Just like how X was input to layer 2.  Each row is a single traning example. (We don't really need this step except for clarity)

A2 = [ones(m, 1) A2]; % adding the bias unit for all training set examples. 5000x26.
Z3 = Theta2 * A2'; % Z3 is a matrix of 10x5000. Each vertical column is z3 for a single training example.
A3 = sigmoid(Z3); % A3 is output matrix of 10x5000.  Each vertical column is output vector for each training example.
H = A3'; % H is 5000x10 where each row is the output vector for a single traning example, corresponds to input X.

% now, y is a 5000x1 vectors each contain the training result in the set [0,1,2,...,num_labels].  
% We want to re-code y so that each element of y is a vector of size num_labels with the approprisate index set to 1
Y = ([1:1:num_labels] == y); % Y is 5000x10 where each row is the re-coded y of a single traing example. 



for i=1 : m % for each training example
   h_i = H(i,:); % prediction for i_th training example 
   y_i = Y(i,:); % training value for i_th training example

   J = J + sum(-1*y_i .* log(h_i) .- (1.-y_i).*log(1 .- h_i ));

end;
% Non-regularized cost computed:
J = J / m;

% Now, compute the regularized term
% Layer 1
Theta1_r = Theta1;
Theta1_r(:,1) = [];
layer1_r = sum(sum(Theta1_r .^2));

Theta2_r = Theta2;
Theta2_r(:,1) = [];
layer2_r = sum(sum(Theta2_r .^2));

J = J + lambda/(2 * m)*(layer1_r + layer2_r);

% Back Propagation

% compute d3 (delta for layer 3) and d2 (delta for layer 2)
Z2_with_bias = [100*ones(1,m);Z2]; %26x5000
DELTA_2 = zeros(size(Theta2));
DELTA_1 = zeros(size(Theta1));
for t=1 : m

   % get input of training example t
   a1 = X(t,:)'; %401x1
   z2 = Theta1 * a1;
   a2 = sigmoid(z2);
   a2 = [1;a2]; % add bias unit
   z3 = Theta2 * a2;
   a3 = sigmoid(z3); % output vector

   % compute d3: 
   y_t = Y'(:,t); % training value for t_th training example.  Note this is a column vector.
   d3 = a3 - y_t; % third layer (output) delta, a column vector 10x1. 
   % compute d2:
   d2 = Theta2' * d3 .* sigmoidGradient([100;z2]); % 26x1

   DELTA_2 = DELTA_2 + d3 * a2';
   DELTA_1 = DELTA_1 + d2(2:end) * a1'; 
end;

% Regularization
% First column corresponds to j=0
reg1 = lambda/m * Theta1_r;
reg1 = [zeros(size(reg1,1),1) reg1]; 
reg2 = lambda/m * Theta2_r;
reg2 = [zeros(size(reg2,1),1) reg2];

Theta1_grad = DELTA_1 / m + reg1;
Theta2_grad = DELTA_2 / m + reg2;











% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
