%test CCT-HTD
addpath(genpath(['.\tensor_toolbox']));
%---------------------Construction of multisource features tensors-------------------------%
str='./MSTAR/2S1_1/';
samples(1).X=heteroFeature_construct(str,1);
str='./MSTAR/T72_1/';
samples(2).X=heteroFeature_construct(str,1);
str='./MSTAR/BRDM_2_1/';
samples(3).X=heteroFeature_construct(str,1);
str='./MSTAR/ZIL_131_1/';
samples(4).X=heteroFeature_construct(str,1);
str='./MSTAR/T62_1/';
samples(5).X=heteroFeature_construct(str,1);
samples_num=zeros(1,size(samples,2));
for i=1:size(samples,2)
    samples_num(i)=size(samples(i).X,2);
end
s(1)=size(samples(1).X{1},1);
s(2)=size(samples(1).X{1},2);
s(3)=size(samples(1).X{1},3);
Xs=tenzeros([s,sum(samples_num)]);
for j=1:size(samples,2)
    for i=1:size(samples(j).X,2)
        Xs(:,:,:,sum(samples_num(1:j-1))+i)=samples(j).X{i}.data;
    end
end
ys=zeros(sum(samples_num),1);
for i=1:size(samples,2)
    ys(sum(samples_num(1:i-1))+1:sum(samples_num(1:i)))=i*ones(samples_num(i),1);
end

str='./MSTAR/2S1_2/';
samples(1).X=heteroFeature_construct(str,1);
str='./MSTAR/T72_2/';
samples(2).X=heteroFeature_construct(str,1);
str='./MSTAR/BRDM_2_2/';
samples(3).X=heteroFeature_construct(str,1);
str='./MSTAR/ZIL_131_2/';
samples(4).X=heteroFeature_construct(str,1);
str='./MSTAR/T62_2/';
samples(5).X=heteroFeature_construct(str,1);
samples_num=zeros(1,size(samples,2));
for i=1:size(samples,2)
    samples_num(i)=size(samples(i).X,2);
end
s(1)=size(samples(1).X{1},1);
s(2)=size(samples(1).X{1},2);
s(3)=size(samples(1).X{1},3);
Xt=tenzeros([s,sum(samples_num)]);
for j=1:size(samples,2)
    for i=1:size(samples(j).X,2)
        Xt(:,:,:,sum(samples_num(1:j-1))+i)=samples(j).X{i}.data;
    end
end
yt=zeros(sum(samples_num),1);
for i=1:size(samples,2)
    yt(sum(samples_num(1:i-1))+1:sum(samples_num(1:i)))=i*ones(samples_num(i),1);
end
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
Id=ceil(s/4);
M=5;
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




