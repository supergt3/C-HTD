function varargout=CCT_HTD(Xs,Xt,ys_,yt_,ys,yt,Id,M,c1,c3)
%%Inputs:
%%%Xs: source data tensor:4D-tensor, and the fourth mode denotes sample mode
%%%Xt: target data tensor,:4D-tensor, and the fourth mode denotes sample mode
%%%Ys_: source label: ns*1
%%%Yt_: target label: nt*1 
%%%Ys: real value of source label: ns*1 
%%%Yt: real value of target labe: nt*1 (only used for testing accuracy)
%%%ld: the dimensions of extracted features
%%%c1: the ratio of removed outliers
%%%c3: regularization parameter
%%%Outputs:
%%%Us,Ut,Gs,Gt,Ws,Wt:the optimization results of CCT-HTD
%%%%ys_,yt_£ºpredicted labels.
%%%%MMD_tr:MMD distance
%%%%acc_seq: the accuracy for each iteration using 1NN
%%%%fval_seq£ºthe value of objective function for each iteration

addpath(genpath(['..\tensor_toolbox']));
Ns=size(Xs,4);
Nt=size(Xt,4);
% M=max(ys_);
L=ndims(Xs)-1;
Is=size(Xs);
It=size(Xt);
%initialize U
Us=cell(1,L+1);
T = tucker_als(Xs,[Id,M]);
for i=1:L
    Us{i}=T.U{i};
%    Us{i}=rand(Is(i),Id(i));
end
Us{L+1}=zeros(Ns,M);
%calculate class mean for source domain
% Xs_cm=cell(1,M);
% for i=1:M
%     temp_v=zeros(Ns,1);
%     temp_v(ys_==i)=1;
%     temp_v=temp_v/size(find(ys_(:)==i),1);
%     Xs_cm{i}=ttv(Xs,temp_v,L+1);
% end
for i=1:Ns
    if ys_(i)~=0
        Us{L+1}(i,ys_(i))=1;
    else
%         Xs_i=Xs(:,:,:,i);
%         temp_dis=zeros(1,M);
%         for j=1:M
%            temp_dis(j)=norm(Xs_i-Xs_cm{j});
%         end
%         [~,ind]=min(temp_dis);
        Us{L+1}(i,:)=rand(1,M);
        Us{L+1}(i,:)=Us{L+1}(i,:)/sum(Us{L+1}(i,:));
    end
end

T = tucker_als(Xt,[Id,M]);
Ut=cell(1,L+1);
for i=1:L
     Ut{i}=T.U{i};
%    Ut{i}=rand(It(i),Id(i));
end
Ut{L+1}=zeros(Nt,M);
%calculate class mean for target domain
% Xt_cm=cell(1,M);
% for i=1:M
%     temp_v=zeros(Nt,1);
%     temp_v(yt_==i)=1;
%     temp_v=temp_v/size(find(yt_(:)==i),1);
%     Xt_cm{i}=ttv(Xt,temp_v,L+1);
% end
for i=1:Nt
    if yt_(i)~=0
        Ut{L+1}(i,yt_(i))=1;
     else
%         Xt_i=Xt(:,:,:,i);
%         temp_dis=zeros(1,M);
%         for j=1:M
%            temp_dis(j)=norm(Xt_i-Xt_cm{j});
%         end
%         [~,ind]=min(temp_dis);
        Ut{L+1}(i,:)=rand(1,M);
        Ut{L+1}(i,:)=Ut{L+1}(i,:)/sum(Ut{L+1}(i,:));
    end
end
%initialize Gs,Gt
Us_tran=cell(1,L+1);
Ut_tran=cell(1,L+1);
for i=1:L
Us_tran{i}=Us{i}';
Ut_tran{i}=Ut{i}';
end
Us_tran{L+1}=(Us{L+1}'*Us{L+1})^-1*Us{L+1}';
Ut_tran{L+1}=(Ut{L+1}'*Ut{L+1})^-1*Ut{L+1}';
Gs=ttm(Xs,{Us_tran{:}},1:L+1);
Gt=ttm(Xt,{Ut_tran{:}},1:L+1);
Gst=(Gs+Gt)/2;
%initialize Ws,Wt
epsilon=c1;
temp_Ws=ones(1,Ns);
temp_Ws(ys_==0)=1*temp_Ws(ys_==0)*sqrt((1-epsilon));
Ws=diag(temp_Ws);
temp_Wt=ones(1,Nt);
temp_Wt(yt_==0)=1*temp_Wt(yt_==0)*sqrt((1-0));
Wt=diag(temp_Wt);
%create Laplace matrix
Lap=zeros(M,M);
for i=1:M
   temp_e=zeros(M,1);
   temp_e(i)=1-1/M;
   temp_e([1:i-1,i+1:M])=-1/M;
   Lap=Lap-c3*(temp_e*temp_e');
end
iter=1;
ind_ys_up=find(ys_==0);
ind_yt_up=find(yt_==0);
Ns_u=size(find(ys_(:)==0),1);
Nt_u=size(find(yt_(:)==0),1);
fval_l=0;
acc_seq=zeros(1,20);
fval_seq=zeros(1,20);
while true
%update U
for i=1:L
Xs2=ttm(Xs,Ws,L+1);
Xs2_mat=tenmat(Xs2,i,'t');
Gs2=ttm(Gst,{Us{[1:i-1,i+1:L+1]}},[1:i-1,i+1:L+1]);
Gs3=ttm(Gs2,Ws,L+1);
Gs3_mat=tenmat(Gs3,i,'t');
temp=tensor(Gs3_mat'*Xs2_mat);
[U,~,V]=svd(temp.data);
Us{i}=V(:,1:size(U,1))*U';

Xt2=ttm(Xt,Wt,L+1);
Xt2_mat=tenmat(Xt2,i,'t');
Gt2=ttm(Gst,{Ut{[1:i-1,i+1:L+1]}},[1:i-1,i+1:L+1]);
Gt3=ttm(Gt2,Wt,L+1);
Gt3_mat=tenmat(Gt3,i,'t');
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
        Gs2=ttm(Gst,{Us{[1:L]}},[1:L]);
        Xs_i_mat=double(reshape(Xs_i,[1,Is(1)*Is(2)*Is(3)]));
        Gs2_mat=double(reshape(Gs2,[Is(1)*Is(2)*Is(3),M]));
        Gs_Xs=Xs_i_mat*Gs2_mat;
        Gs_Gs=Gs2_mat'*Gs2_mat;
%         Us_i=ADMM_CH_Tucker(Gs2_mat,Xs_i_mat,Gs_Xs,Gs_Gs);
%         [0,0,0,0,1]*0.25*Gs_Gs/Gs_Gs_norm*[0,0,0,0,1]'-Gs_Xs/Gs_Gs_norm*[0,0,0,0,1]'
        Gs_Gs_norm=norm(Gs_Gs);
        options = optimoptions('quadprog','Algorithm','interior-point-convex','OptimalityTolerance',10^-12,'MaxIterations',1000);%'Display','iter', 'sqp','MaxIterations',1000000,'MaxFunctionEvaluations',1000000
        [Us_i,fval,exitflag]=quadprog(Gs_Gs/Gs_Gs_norm,-Gs_Xs/Gs_Gs_norm,[],[],ones(1,M),1,zeros(1,M),[],[],options );
%         ones(1,M)*alpha
        Us{L+1}(i,:)=Us_i';
    end
end
for i=1:Nt
    %ADMM method
    if yt_(i)==0
        temp_v=zeros(Nt,1);
        temp_v(i)=1;
        Xt_i=ttv(Xt,temp_v,L+1);
        Gt2=ttm(Gst,{Ut{[1:L]}},[1:L]);
        Xt_i_mat=double(reshape(Xt_i,[1,It(1)*It(2)*It(3)]));
        Gt2_mat=double(reshape(Gt2,[It(1)*It(2)*It(3),M]));
        Gt_Xt=Xt_i_mat*Gt2_mat;
        Gt_Gt=Gt2_mat'*Gt2_mat;
%         Ut_i=ADMM_CH_Tucker(Gt2_mat,Xt_i_mat,Gt_Xt,Gt_Gt);
        Gt_Gt_norm=norm(Gt_Gt);
        options = optimoptions('quadprog','Algorithm','interior-point-convex','OptimalityTolerance',10^-12,'MaxIterations',1000);%'Display','iter', 'sqp','MaxIterations',1000000,'MaxFunctionEvaluations',1000000
        [Ut_i,fval,exitflag]=quadprog(Gt_Gt/Gt_Gt_norm,-Gt_Xt/Gt_Gt_norm,[],[],ones(1,M),1,zeros(1,M),[],[],options );
        Ut{L+1}(i,:)=Ut_i';
    end
end
% for i=1:Ns
%     if ys(i)==0
%         temp_v=zeros(Ns,1);
%         temp_v(i)=1;
%         Xs_i=ttv(Xs,temp_v,L+1);
%         Gs2=ttm(Gs,{Us{[1:L]}},[1:L]);
%         temp_dis=zeros(1,M);
%         for j=1:M
%             temp_Gs_j=Gs2(:,:,:,j);
%             temp_dis(j)=norm(Xs_i-temp_Gs_j);
%         end
%         [~,ind]=min(temp_dis);
%         Us{L+1}(i,:)=0;
%         Us{L+1}(i,ind)=1;
%     end
% end
% for i=1:Nt
%     if yt(i)==0
%         temp_v=zeros(Nt,1);
%         temp_v(i)=1;
%         Xt_i=ttv(Xt,temp_v,L+1);
%         Gt2=ttm(Gt,{Ut{[1:L]}},[1:L]);
%         temp_dis=zeros(1,M);
%         for j=1:M
%             temp_Gt_j=Gt2(:,:,:,j);
%             temp_dis(j)=norm(Xt_i-temp_Gt_j);
%         end
%         [~,ind]=min(temp_dis);
%         Ut{L+1}(i,:)=0;
%         Ut{L+1}(i,ind)=1;
%     end
% end
%update Gs,Gt
for count=1:1:1
for i=1:L
Us_tran{i}=Us{i}';
Ut_tran{i}=Ut{i}';
end
Xs_p=ttm(Xs,{Us_tran{1:L}},[1:L]);
Xt_p=ttm(Xt,{Ut_tran{1:L}},[1:L]);
Xs_w=ttm(Xs_p,Ws,L+1);
Xt_w=ttm(Xt_p,Wt,L+1);
Xs2=tenmat(Xs_w,L+1,'t');
Xt2=tenmat(Xt_w,L+1,'t');
Ws_U=Ws*Us{L+1};
H_Ws_U=Ws_U'*Ws_U;
Wt_U=Wt*Ut{L+1};
H_Wt_U=Wt_U'*Wt_U;
Gs_2=permute(Gst,[4,1,2,3]);
Gs_3=double(reshape(Gs_2,[M,Id(1)*Id(2)*Id(3)]));
G_tol=[Gs_3];
for i=1:M
    temp_Ws_U=Ws_U(:,i);
    temp_Wt_U=Wt_U(:,i);
    temp_L=Lap(i,:);
    temp_L(i)=0;
    temp_L2=H_Ws_U(i,:)+H_Wt_U(i,:);
    temp_L2(i)=0;
    Gs_new=(H_Ws_U(i,i)+H_Wt_U(i,i)+Lap(i,i))^-1*(double(Xs2)*temp_Ws_U+double(Xt2)*temp_Wt_U-G_tol'*(temp_L+temp_L2)');
%     Gs_new=(Lap(i,i))^-1*(-G_tol'*temp_L');
    Gs_new_t=tensor(Gs_new,Id);
    Gs(:,:,:,i)=Gs_new_t;
    G_tol(i,:)=Gs_new';
%     Gs_new=2*(temp_Ws_U'*temp_Ws_U+Lap(i,i))^-1*(double(tensor(Hs'*Xs2*temp_Ws_U))-G_tol'*temp_L');
    
end
end
%update Ws,Wt
if ~isempty(find(ys_==0))
Gs2=ttm(Gst,{Us{:}},1:L+1);
temp_dis=zeros(1,Ns);
for i=1:Ns
    temp_v=zeros(Ns,1);
    temp_v(i)=1;
    Gs_i=ttv(Gs2,temp_v,L+1);
    Xs_i=ttv(Xs,temp_v,L+1);
    temp_dis(i)=norm(Gs_i-Xs_i);
end
[~,ind]=sort(temp_dis(ys_==0));
% [~,ind]=sort(temp_dis);
temp_1=zeros(1,Ns);
% temp_1(ind(1:round((1-epsilon)*Nsu)))=1;
temp_1(ys_~=0)=1;
temp_1(ind_ys_up(ind(1:round((1-epsilon)*Ns_u))))=1;
temp_1(ys_==0)=1*temp_1(ys_==0);
Ws=diag(temp_1);
end
if ~isempty(find(yt_==0))
Gt2=ttm(Gst,{Ut{:}},1:L+1);
temp_dis=zeros(1,Nt);
for i=1:Nt
    temp_v=zeros(Nt,1);
    temp_v(i)=1;
    Gt_i=ttv(Gt2,temp_v,L+1);
    Xt_i=ttv(Xt,temp_v,L+1);
    temp_dis(i)=norm(Gt_i-Xt_i);
end
[~,ind]=sort(temp_dis(yt_==0));
temp_1=zeros(1,Nt);
temp_1(yt_~=0)=1;
temp_1(ind_yt_up(ind(1:round((1-0)*Nt_u))))=1;
temp_1(yt_==0)=1*temp_1(yt_==0);
Wt=diag(temp_1);
end
%calculate objective function value
Xs_w=ttm(Xs,Ws,L+1);
Gs2=ttm(Gst,{Us{:}},[1:L+1]);
Gs3=ttm(Gs2,Ws,L+1);
temp1=norm(Gs3-Xs_w)^2;
Xt_w=ttm(Xt,Wt,L+1);
Gt2=ttm(Gst,{Ut{:}},[1:L+1]);
Gt3=ttm(Gt2,Wt,L+1);
temp2=norm(Gt3-Xt_w)^2;
Gs_2=permute(Gst,[4,1,2,3]);
Gs_3=double(reshape(Gs_2,[M,Id(1)*Id(2)*Id(3)]));
temp3=trace(Gs_3'*Lap*Gs_3);
fval=temp1+temp2+temp3
MMD_tr=temp2;
if iter>6||abs(fval-fval_l)<0.01
    break;
else
    fval_l=fval;
end
iter=iter+1;
for i=1:L
Us{i}=Us{i}';
Ut{i}=Ut{i}';
end
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
%1NN
% ind_W=diag(Ws);
Ws_d=diag(Ws);
% Mdl = fitcknn(Xs_(:,Ws_d==1)',ys(Ws_d==1),'NumNeighbors',1);
Mdl = fitcknn(Xs_',ys,'NumNeighbors',1);
Label_p = predict(Mdl,Xt_');
acc_seq(iter)=size(find(Label_p==yt),1)/size(yt,1);
fval_seq(iter)=fval;
for i=1:L
Us{i}=Us{i}';
Ut{i}=Ut{i}';
end
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
if nargout==6
    varargout={Us,Ut,Gs,Gt,Ws,Wt};
end
if nargout==8
    [~,ind]=max(Us{L+1},[],2);
    ys_=ind;
    [~,ind]=max(Ut{L+1},[],2);
    yt_=ind;
    varargout={Us,Ut,Gs,Gt,Ws,Wt,ys_,yt_};
end
if nargout==9
    [~,ind]=max(Us{L+1},[],2);
    ys_=ind;
    [~,ind]=max(Ut{L+1},[],2);
    yt_=ind;
    varargout={Us,Ut,Gs,Gt,Ws,Wt,ys_,yt_,MMD_tr};
end
if nargout==11
    [~,ind]=max(Us{L+1},[],2);
    ys_=ind;
    [~,ind]=max(Ut{L+1},[],2);
    yt_=ind;
    varargout={Us,Ut,Gs,Gt,Ws,Wt,ys_,yt_,MMD_tr,acc_seq,fval_seq};
end
end