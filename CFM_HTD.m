function varargout=CFM_HTD(Xs,Xt,ys_,ys,Id,M,c2)
%%Inputs:
%%%Xs: source data tensor:4D-tensor, and the fourth mode denotes sample mode
%%%Xt: target data tensor:4D-tensor, and the fourth mode denotes sample mode
%%%Ys_: class label: ns*1
%%%Ys: real value of class label: ns*1 
%%%ld: the dimensions of extracted features
%%%M:the number of classes
%%%c2: regularization parameter
%%%Outputs:
%%%Us,Ut,Gs,Gt,Ws,Wt:the optimization results of CCT-HTD
%%%%ys_£ºpredicted labels.
%%%%acc_seq: the accuracy for each iteration using 1NN
%%%%fval_seq£ºthe value of objective function for each iteration
addpath(genpath(['..\tensor_toolbox']));
Ns=size(Xs,4);
% M=max(ys_);
L=ndims(Xs)-1;
Is=size(Xs);
It=size(Xt);
%initialize U
Us=cell(1,L+1);
for i=1:L
   Us{i}=rand(Is(i),Id(i));
end
Us{L+1}=zeros(Ns,M);
for i=1:Ns
    if ys_(i)~=0
        Us{L+1}(i,ys_(i))=1;
    else
        Us{L+1}(i,:)=rand(1,M);
        Us{L+1}(i,:)=Us{L+1}(i,:)/sum(Us{L+1}(i,:));
    end
end
Ut=cell(1,L+1);
for i=1:L
   Ut{i}=rand(It(i),Id(i));
end
Ut{L+1}=Us{L+1};
%initialize Gs,Gt
Us_tran=cell(1,L+1);
Ut_tran=cell(1,L+1);
for i=1:L
Us_tran{i}=Us{i}';
Ut_tran{i}=Ut{i}';
end
Us_tran{L+1}=(Us{L+1}'*Us{L+1})^-1*Us{L+1}';
Ut_tran{L+1}=Us_tran{L+1};
Gs=ttm(Xs,{Us_tran{:}},1:L+1);
Gt=ttm(Xt,{Ut_tran{:}},1:L+1);
%create Laplace matrix
Lap=zeros(2*M,2*M);
for i=1:M
   temp_e=zeros(2*M,1);
   temp_e(i)=1-1/M;
   temp_e([1:i-1,i+1:M])=-1/M;
   Lap=Lap-c2*(temp_e*temp_e');
   temp_e=zeros(2*M,1);
   temp_e(i+M)=1-1/M;
   temp_e([M+1:M+i-1,M+i+1:2*M])=-1/M;
   Lap=Lap-c2*(temp_e*temp_e');
end
iter=1;
fval_l=0;
fval_seq=[]
while true
%update U
for i=1:L
Xs2_mat=tenmat(Xs,i,'t');
Gs2=ttm(Gs,{Us{[1:i-1,i+1:L+1]}},[1:i-1,i+1:L+1]);
Gs3_mat=tenmat(Gs2,i,'t');
temp=tensor(Gs3_mat'*Xs2_mat);
[U,~,V]=svd(temp.data);
Us{i}=V(:,1:size(U,1))*U';

Xt2_mat=tenmat(Xt,i,'t');
Gt2=ttm(Gt,{Ut{[1:i-1,i+1:L+1]}},[1:i-1,i+1:L+1]);
Gt3_mat=tenmat(Gt2,i,'t');
temp=tensor(Gt3_mat'*Xt2_mat);
[U,~,V]=svd(temp.data);
Ut{i}=V(:,1:size(U,1))*U';
end
%update Us{L+1} Ut{L+1}
for i=1:Ns
    %ADMM method
    if ys_(i)==0
        temp_v=zeros(Ns,1);
        temp_v(i)=1;
        Xs_i=ttv(Xs,temp_v,L+1);
%         temp_v=zeros(Ns,1);
%         temp_v(2)=1;
%         Xs_i2=ttv(Xs,temp_v,L+1)        
        Gs2=ttm(Gs,{Us{[1:L]}},[1:L]);
        Xs_i_mat=double(reshape(Xs_i,[1,Is(1)*Is(2)*Is(3)]));
        Gs2_mat=double(reshape(Gs2,[Is(1)*Is(2)*Is(3),M]));
        Gs_Xs=Xs_i_mat*Gs2_mat;
        Gs_Gs=Gs2_mat'*Gs2_mat;
        
        Xt_i=ttv(Xt,temp_v,L+1);  
        Gt2=ttm(Gt,{Ut{[1:L]}},[1:L]);
        Xt_i_mat=double(reshape(Xt_i,[1,It(1)*It(2)*It(3)]));
        Gt2_mat=double(reshape(Gt2,[It(1)*It(2)*It(3),M]));
        Gt_Xt=Xt_i_mat*Gt2_mat;
        Gt_Gt=Gt2_mat'*Gt2_mat;
%         Us_i=ADMM_CH_Tucker(Gs2_mat,Xs_i_mat,Gs_Xs,Gs_Gs);
        Gs_Gs_norm=norm(Gs_Gs);
        options = optimoptions('quadprog','Algorithm','interior-point-convex','OptimalityTolerance',10^-12,'MaxIterations',1000);%'Display','iter', 'sqp','MaxIterations',1000000,'MaxFunctionEvaluations',1000000
        [Us_i,fval,exitflag]=quadprog((Gs_Gs+Gt_Gt)/Gs_Gs_norm,-(Gs_Xs+Gt_Xt)/Gs_Gs_norm,[],[],ones(1,M),1,zeros(1,M),[],[],options );
%         ones(1,M)*alpha
        Us{L+1}(i,:)=Us_i';
        Ut{L+1}(i,:)=Us_i';
    end
end
%update Gs,Gt
for i=1:L
Us_tran{i}=Us{i}';
Ut_tran{i}=Ut{i}';
end
Xs_p=ttm(Xs,{Us_tran{1:L}},[1:L]);
Xt_p=ttm(Xt,{Ut_tran{1:L}},[1:L]);
Xs2=tenmat(Xs_p,L+1,'t');
Xt2=tenmat(Xt_p,L+1,'t');
Ws_U=Us{L+1};
H_Ws_U=Ws_U'*Ws_U;
Gs_2=permute(Gs,[4,1,2,3]);
Gs_3=double(reshape(Gs_2,[M,Id(1)*Id(2)*Id(3)]));
Gt_2=permute(Gt,[4,1,2,3]);
Gt_3=double(reshape(Gt_2,[M,Id(1)*Id(2)*Id(3)]));
G_tol=[Gs_3;Gt_3];
for i=1:M
    temp_Ws_U=Ws_U(:,i);
    temp_L=Lap(i,:);
    temp_L(i)=0;
    temp_L2=H_Ws_U(i,:);
    temp_L2(i)=0;
    temp_L2=[temp_L2,zeros(1,M)];
    Gs_new=(H_Ws_U(i,i)+Lap(i,i))^-1*(double(Xs2)*temp_Ws_U-G_tol'*(temp_L2+temp_L)');
%     Gs_new=(Lap(i,i))^-1*(-G_tol'*temp_L');
    Gs_new_t=tensor(Gs_new,Id);
    Gs(:,:,:,i)=Gs_new_t;
    G_tol(i,:)=Gs_new';
%     Gs_new=2*(temp_Ws_U'*temp_Ws_U+Lap(i,i))^-1*(double(tensor(Hs'*Xs2*temp_Ws_U))-G_tol'*temp_L');
    
end
Wt_U=Ut{L+1};
H_Wt_U=Wt_U'*Wt_U;
for i=M+1:2*M
    temp_Wt_U=Wt_U(:,i-M);
    temp_L=Lap(i,:);
    temp_L(i)=0;
    temp_L2=H_Wt_U(i-M,:);
    temp_L2(i-M)=0;
    temp_L2=[zeros(1,M),temp_L2];
    Gt_new=(H_Wt_U(i-M,i-M)+Lap(i,i))^-1*(double(Xt2)*temp_Wt_U-G_tol'*(temp_L2+temp_L)');
%     Gt_new=(Lap(i,i))^-1*(-G_tol'*temp_L');
    Gt_new_t=tensor(Gt_new,Id);
    Gt(:,:,:,i-M)=Gt_new_t;
     G_tol(i,:)=Gt_new';
%     Gs_new=2*(temp_Ws_U'*temp_Ws_U+Lap(i,i))^-1*(double(tensor(Hs'*Xs2*temp_Ws_U))-G_tol'*temp_L');
end

%calculate objective function value
Gs2=ttm(Gs,{Us{:}},[1:L+1]);
temp1=norm(Gs2-Xs)^2;
Gt2=ttm(Gt,{Ut{:}},[1:L+1]);
temp2=norm(Gt2-Xt)^2;
fval=temp1+temp2;
iter=iter+1;
if iter>5||abs(fval-fval_l)<0.01
    break;
else
    fval_l=fval;
    fval_seq=[fval_seq,fval];
end
%predict label of samples
[~,ind]=max(Us{L+1},[],2);
acc_seq(iter)=size(find(ind==ys),1)/size(ys(:),1);
end
for i=1:L
Us{i}=Us{i}';
Ut{i}=Ut{i}';
end
if nargout==2
    varargout={Us,Ut};
end
if nargout==4
    varargout={Us,Ut,Gs,Gt};
end
if nargout==8
    [~,ind]=max(Us{L+1},[],2);
    ys_=ind;
    [~,ind]=max(Ut{L+1},[],2);
    yt_=ind;
    varargout={Us,Ut,Gs,Gt,ys_,yt_,acc_seq,fval_seq};
end
end