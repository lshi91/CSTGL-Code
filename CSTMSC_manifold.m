% function [result]=CSTMSC(X,alpha,beta,omega,gt)
% function [result,S1,S2,S_aug,err_S] = CSTMSC_manifold(X,alpha,beta,gamma,gt)
function [result,S1,S2,S_aug,err_S] = CSTMSC_manifold(X,alpha,beta,gamma,gt)
Cluster_num = size(unique(gt),1);
V = size(X,2); %number of views
N = size(X{1},2);% number of data points
NITER = 20;
% unit norm
for i=1:V
    X{i} = X{i}./(repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1)+eps);
end
% MinMax norm
% for i = 1:V
%     dist = max(max(X{i})) - min(min(X{i}));
%     X{i} = (X{i} - min(min(X{i})))/dist;  % MinMax归一化至[0,1]
% end
% BN norm
% for i = 1:V
%     X{i} = mapstd(X{i},0,1); % 数据预处理--Batch Norm
% end
% for i = 1:V
%     X{i} = transpose(mapstd(X{i}',0,1)); % 数据预处理--Layer Norm
% end
%% Initilize A, W, K, S1, S2, E, Q1, Q2, Q3,
sX = [N,N,V];
for i = 1:V
    % subproblems
    A{i} = constructW_PKN(X{i},10);  
    W{i} = zeros(N,N);
    K{i} = zeros(N,N); 
    S1{i} = zeros(N,N); 
    S2{i} = zeros(N,N);
    E{i} = zeros(N,N);
    sd{i} = L2_distance_1(X{i}, X{i});  % euclidean 
    % sd{i} = stiefel_distance(X{i});  %stiefel manifold geodesic distance
    % Lagrange multipliers
    Q1{i} = zeros(N,N);
    Q2{i} = zeros(N,N); 
    Q3{i} = zeros(N,N);
end

rho = 0.01; 
mu = 2;

%% outer loop
for iter = 1:NITER
    %% == update A{i} ==
    for i = 1:V
        temp_A = zeros(N);
        B = S1{i} + S2{i} + E{i} - Q1{i}/rho;
        tmp_sd = sd{i};
        for j = 1:N
            ad = (rho*B(j,:)-tmp_sd(j,:))/(2*alpha+rho);
            temp_A(j,:) = EProjSimplex_new(ad);
        end
        A{i} = temp_A;
    end

    %% == update W{i} ==
    C = [];
    for i = 1:V
        slice = E{i}+Q3{i}/rho;
        C = [C; slice(:,:)];
    end
    [Wconcat] = solve_l1l2(C, 1/rho);
    for i = 1:V
        W{i} = Wconcat((i-1)*N+1:i*N,:);
    end

    %% == update K ==
    S1_tensor = cat(3, S1{:,:});
    Q2_tensor = cat(3, Q2{:,:});
    s1 = S1_tensor(:);
    q2 = Q2_tensor(:);
    [j, ~] = wshrinkObj(s1 + 1/rho*q2,beta/rho,sX,0,3);
    K_tensor = reshape(j, sX);
    for i = 1:V
        K{i} = K_tensor(:,:,i);
    end
    
    %% == update S2{i} ==
    A_tensor = cat(3, A{:,:}); % construct tensor from A{i}
    Ahat_tensor = fft(A_tensor,[],3); 
    Q1_tensor = cat(3, Q1{:,:}); % construct tensor from Q1{i}
    Q1hat_tensor = fft(Q1_tensor,[],3);
    S1_tensor = cat(3, S1{:,:}); % construct tensor from S1{i}
    S1hat_tensor = fft(S1_tensor,[],3);
    E_tensor = cat(3, E{:,:}); % construct tensor from E{i}
    Ehat_tensor = fft(E_tensor,[],3);
    for i = 1:V
        S2hat_tensor(:,:,i) = rho*(Ahat_tensor(:,:,i)+Q1hat_tensor(:,:,i)/rho-S1hat_tensor(:,:,i)-Ehat_tensor(:,:,i))/(2*gamma+rho);
    end
    S2_tensor = ifft(S2hat_tensor,[],3);
    for i = 1:V
        S2{i} = S2_tensor(:,:,i);
    end

    %% == update S1{i} ==
    S_old = S1_tensor;
    K_tensor = cat(3, K{:,:}); % construct tensor from K{i}
    Khat_tensor = fft(K_tensor,[],3); 
    S2_tensor = cat(3, S2{:,:}); % construct tensor from S2{i}
    S2hat_tensor = fft(S2_tensor,[],3);
    Q2_tensor = cat(3, Q2{:,:}); % construct tensor from Q2{i}
    Q2hat_tensor = fft(Q2_tensor,[],3);
    for i = 1:V
        S1hat_tensor(:,:,i) = (Khat_tensor(:,:,i)-Q2hat_tensor(:,:,i)/rho-(S2hat_tensor(:,:,i)+Ehat_tensor(:,:,i)...
                                 -(Ahat_tensor(:,:,i)+Q1hat_tensor(:,:,i)/rho)))/2;
    end
    S1_tensor = ifft(S1hat_tensor,[],3);
    for i = 1:V
        S1{i} = S1_tensor(:,:,i);
    end

    %% == update E{i} ==
    W_tensor = cat(3, W{:,:}); % construct tensor from W{i}
    What_tensor = fft(W_tensor,[],3);
    S1_tensor = cat(3, S1{:,:}); % construct tensor from S1{i}
    S1hat_tensor = fft(S1_tensor,[],3);
    % S2_tensor = cat(3, S2{:,:}); % construct tensor from S2{i}
    % S2hat_tensor = fft(S2_tensor,[],3);
    Q3_tensor = cat(3, Q3{:,:}); % construct tensor from Q3{i}
    Q3hat_tensor = fft(Q3_tensor,[],3);
    for i = 1:V
        Ehat_tensor(:,:,i) = (Ahat_tensor(:,:,i)+Q1hat_tensor(:,:,i)/rho+What_tensor(:,:,i)...
            -S1hat_tensor(:,:,i)-S2hat_tensor(:,:,i)-Q3hat_tensor(:,:,i)/rho)/2;
    end
    E_tensor = ifft(Ehat_tensor,[],3);
    for i = 1:V
        E{i} = E_tensor(:,:,i);
    end
    % E_size = size(Ehat_tensor);
    % A_size = size(Ahat_tensor);
    
   %% == update Q1{i}, Q2, Q3{i} ==
    for i = 1:V
        Q1{i} = Q1{i} + rho*(A{i}-(S1{i}+S2{i})-E{i});
        Q2{i} = Q2{i} + rho*(S1{i}-K{i});
        Q3{i} = Q3{i} + rho*(E{i}-W{i});
    end
    % Q2 = Q2 + rho*(S-K);
    
    %% == update rho ==
    rho = rho*mu;
    % record err
    err_S(iter) = norm(S1_tensor-S_old,"fro");
    % if iter > 3 && err_S(iter) < 1e-2
    %     break
    % end
end
% fprintf('迭代%d次\n',iter);
%% perform clustering
S_aug = zeros(N,N);
for j = 1:V
    S_aug = S_aug + (S1{j}+S2{j})/V;
end
S_aug = (S_aug+S_aug')/2;
S_aug = S_aug - diag(diag(S_aug));
S_aug = S_aug ./ (repmat(sqrt(sum(S_aug.^2,1)),size(S_aug,1),1)+eps);  % 单位向量归一化
y = spectral_clustering(S_aug,Cluster_num);
Y = y';
%% perform clustering v2
% S_aug = zeros(N,N);
% S = (S + S') / 2;  % 对称
% S = S - diag(diag(S));  % 去对角
% S = S ./ (repmat(sqrt(sum(S.^2,1)),size(S,1),1)+eps);  % 单位化
% for j = 1:V
%     S2{j} = (S2{j} + S2{j}') / 2;
%     S2{j} = S2{j} - diag(diag(S2{j}));
%     S2{j} = S2{j} ./ (repmat(sqrt(sum(S2{j}.^2,1)),size(S2{j},1),1)+eps);
%     S_aug = S_aug + S2{j};
% end
% S_aug = S_aug + S;
% S_aug = (S_aug + S_aug')/2;
% S_aug = S_aug - diag(diag(S_aug));
% Y = spectralcluster(S_aug,Cluster_num,"Distance","precomputed");

%% 计算指标
[ACC,NMI,PUR] = ClusteringMeasure(gt,Y); %ACC NMI Purity
[Fscore,Precision,R] = compute_f(gt,Y);
[AR,~,~,~]=RandIndex(gt,Y);
result = [ACC NMI AR Fscore Precision R PUR];

