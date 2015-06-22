function [ dict ] = trdict_preprocess( dict_pre , dict_label ,Tr_Num )
%dict preprocessing
%prior_cal;
dict=cell(Tr_Num,1);
for i=0:Tr_Num-1
    dict{i+1,1}=dict_pre(find(dict_label==i+1),:);
end
end
