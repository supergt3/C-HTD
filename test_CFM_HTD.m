%test CFM-HTD
addpath(genpath(['.\tensor_toolbox']));
load CFM-HTD.mat %Xs,Xt,ys,yt denote the samples from different sources and the corresponding labels
%set the parameter of CFM-HTD
M=5;%the number of classes
sample_rate=[0.2:0.1:1];
Id=ceil(s/6);%the dimension of extracted features;
r_num=rand(1,size(Xs,4));
[~,ind]=sort(r_num);
ys_=ys;
ys_(ind(round(size(Xs,4)*sample_rate(4)):end))=0;%the ratio of unlabeled samples is set to 0.5
r_num=rand(1,size(Xt,4));
[~,ind]=sort(r_num);
c2=0.25;
tic;
[Us,Ut,Gs,Gt,ys1,yt1,acc_seq,fval_seq]=CFM_HTD(Xs,Xt,ys_,ys,Id,M,c2);%perform CFM-HTD
toc;
acc_seq(end) %accuracy
%calculate fused features
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
X=[Xs_;Xt_];
for i=1:size(samples_num,2)
    X_fusion(i).X=X(:,sum(samples_num(1:i-1))+1:sum(samples_num(1:i)));
end
NMI_v=zeros(1,5);
for k=1:5
Idx=kmeans([Xs_;Xt_]',5);
NMI_v(k)=NMI(Idx,ys);
end
NMI_set=max(NMI_v);%calculate NMI




