% =========================================================================
% An example code for the kernel Locality-preserving K-SVD algorithm
%
%   Weiyang Liu, Zhiding Yu, Meng Yang, Lijia Lu, and Yuexian Zou.
%   "Joint Kernel Dictionary and Classifier Learning for Sparse Coding via  
%    Locality Preserving K-SVD", ICME 2015.
%
% Author: Weiyang Liu (wyliu@pku.edu.cn)
% Date: 2014.11.10
% =========================================================================

clear all;
clc;
addpath(genpath('.\ksvdbox'));  % add K-SVD box
addpath(genpath('.\OMPbox')); % add OMP box
load('.\trainingdata\featurevectors.mat','training_feats', 'testing_feats', 'H_train', 'H_test'); % extended yaleB dataset
%% model parameter settings
sparsitythres = 300; % sparsity prior
sqrt_alpha = 3; % weights for locality preserving term
sqrt_beta = 7; % weights for classification err term
local_base = 50; % local bases number
dictsize = 1216; % dictionary size
iterations = 50; % iteration number
iterations4ini = 30; % iteration number for initialization
outer_iter = 10; % iteration number for learning the locality preserving matrix
kernel_sign = 1; % enable the LP-KSVD to learn in the kernel space
%% learning in kernel space
if kernel_sign
    training_feats_m=double(training_feats);
    testing_feats=double(testing_feats);
    training_feats_m=double(training_feats*diag(1./sqrt(sum(training_feats.*training_feats))));
%% kernelize the training samples
    K=[];
    K=double(K);
    for i=1:size(training_feats_m,2)
        for j=1:size(training_feats_m,2)
            K(i,j)=double(kernel(training_feats_m(:,i),training_feats_m(:,j)));
        end
    end
    dict_label=zeros(1,size(H_train,2));
    for i=1:size(H_train,2)
        dict_label(1,i)=find(H_train(:,i)==1);
    end
    Tr_Num=size(H_train,1);
    [ K2 ] = trdict_preprocess( K , dict_label ,Tr_Num );
    K_fin=[];
    K_m=[];
    for i=1:size(K2,1)
        K_m=cell2mat(K2(i,1));
        K_fin=[K_fin;K_m];
    end
    K=K_fin;
    K=K';
    training_feats=K;
%% kernelize the testing samples
    testing_feats_m=zeros(size(testing_feats));
    testing_feats_m=double(testing_feats_m);
    testing_feats_m=double(testing_feats*diag(1./sqrt(sum(testing_feats.*testing_feats))));
    K=[];
    for i=1:size(training_feats,2)
        for j=1:size(testing_feats,2)
            K(i,j)=double(kernel(training_feats_m(:,i),testing_feats_m(:,j)));
        end
    end
    K=K;
    testing_feats=K;
end
%%
i=1; % reset i as 1
%% dictionary learning stage
Q_in=[];
for j=1:outer_iter
    if j==1
        s_alpha=0;
    else
        s_alpha=sqrt_alpha;
    end
%%
% get initial dictionary Dinit and Winit
    fprintf('\nLocality Preserving K-SVD initialization - iteration %d', j);
    [Dinit,Tinit,Winit,Q_train] = initialization4LPKSVD(training_feats,H_train,dictsize,iterations4ini,sparsitythres,Q_in,j);
    fprintf('\nCompleted!');
    Q_cum{i}=Q_train;
    i=i+1;
%%
% run LP k-svd training
    fprintf('\nJoint Dictionary and classifier learning via LP K-SVD - iteration %d', j)
    [D2,X2,T2,W2] = localitypreservingKSVD(training_feats,Dinit,Q_train,Tinit,H_train,Winit,iterations,sparsitythres,s_alpha,sqrt_beta);
    % save('.\trainingdata\dictionarydata.mat','D2','X2','W2','T2');
    fprintf('\nCompleted!');
%%
    Q_in=zeros(size(Q_train)); % initialize the locality prserving matrix with zero matrix
% finding local bases
    for frameid=1:size(D2,2)
        [dists,neighbors] = top_K_neighbors( D2,training_feats(:,frameid), local_base );
        Q_in(neighbors,frameid)=1;
    end
end
% save('.\trainingdata\LPmatrixData.mat','Q_cum');
%% classification stage
[prediction,accuracy] = classification(D2, W2, testing_feats, H_test, sparsitythres);
fprintf('\nFinal recognition rate for LP-KSVD is : %.03f ', accuracy);
%% 