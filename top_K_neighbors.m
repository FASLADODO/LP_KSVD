function [dists,neighbors] = top_K_neighbors( X_train,X_test,K )
% Written by Weiyang Liu
%   Input: 
%   X_test the test vector with P*1
%   X_train and y_train are the train data set
%   K is the K neighbor parameter
[~, N_train] = size(X_train);
X_train=X_train*diag(1./sqrt(sum(X_train.*X_train)));
X_test=X_test/norm(X_test,2);
test_mat = repmat(X_test,1,N_train);
dist_mat = (X_train-double(test_mat)) .^2;
% dist_mat = (X_train.*double(test_mat)) ;
% The distance is the Euclid Distance.
dist_array = sum(dist_mat);
[dists, neighbors] = sort(dist_array,'descend');
% The neighbors are the index of top K nearest points.
dists = dists(1:K);
neighbors = neighbors(1:K);
end