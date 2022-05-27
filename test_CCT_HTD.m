%test CCT-HTD
addpath(genpath(['.\tensor_toolbox']));
load CCT-HTD.mat %Xs,Xt,ys,yt denote the samples from different sources and the corresponding labels
%set the parameter of CCT-HTD
sample_rate=[0.2:0.1:1];
r_num=rand(1,size(Xs,4));
[~,ind]=sort(r_num);
ys_=ys;
ys_(ind(round(size(Xs,4)*sample_rate(4)):end))=0;
r_num=rand(1,size(Xt,4));
[~,ind]=sort(r_num);
yt_=yt;
yt_(ind(round(size(Xt,4)*sample_rate(4)):end))=0;
Id=ceil(s/4);%the dimension of extracted features;
M=5;%the number of classes
c1=0.2;
c3=0.5;
tic;
[Us,Ut,Gs,Gt,Ws,Wt,ys1,yt1,MMD_tr,acc_seq,fval_seq]=CCT_HTD(Xs,Xt,ys_,yt_,ys,yt,Id,M,c1,c3);%perform CCT-HTD
toc;
ys_com=[ys_,ys1];
yt_com=[yt_,yt1];
%compute the transferred results
Xs_=zeros(Id(1)*Id(2)*Id(3),size(Xs,4));
for i=1:size(Xs,4)
   Xs_i=Xs(:,:,:,i);
   Xs_i_=ttm(Xs_i,{Us{1:3}},[1:3]);
   Xs_(:,i)=Xs_i_.data(:);
end
Xt_=zeros(Id(1)*Id(2)*Id(3),size(Xt,4));
for i=1:size(Xt,4)
   Xt_i=Xt(:,:,:,i);
   Xt_i_=ttm(Xt_i,{Ut{1:3}},[1:3]);
   Xt_(:,i)=Xt_i_.data(:);
end
%classification using 1NN
Mdl = fitcknn(Xs_',ys,'NumNeighbors',1);
Label_p = predict(Mdl,Xt_');
acc=size(find(Label_p==yt),1)/size(yt,1);




