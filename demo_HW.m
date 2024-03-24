%% 加载数据集
clear;
addpath('Datasets/');
load('HW2.mat');
gt = truth';
%% 运行CSTMSC_manifold
alpha = 1e-1; 
beta = 1e2;
gamma = 1e3;

tic
[result,S1,S2,S_aug,err_S] = CSTMSC_manifold(X,alpha,beta,gamma,gt);
timer= toc;
fprintf('Runtime: %.2f \n ACC: %.4f \t NMI: %.4f \t ARI: %.4f \t Fscore: %.4f\n',[timer result(1) result(2) result(3) result(4)]);
%% 可视化
% subplot(2,3,1)
% image(S1{1},'CDataMapping','scaled')
% colorbar
% title('S1{1}')
% subplot(2,3,2)
% image(S1{2},'CDataMapping','scaled')
% colorbar
% title('S1{2}')
% subplot(2,3,3)
% image(S2{1},'CDataMapping','scaled')
% colorbar
% title('S2{1}')
% subplot(2,3,4)
% image(S2{2},'CDataMapping','scaled')
% colorbar
% title('S2{2}')
% subplot(2,3,5)
% image(S_aug,'CDataMapping','scaled')
% colorbar
% title('S aug')
% colormap("default")
% subplot(2,3,6)
% plot(err_S)
% title('Convergence')
%% 调参
% result_final = zeros(7,7,7);
% para_range = [1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3];
% for i = 1:length(para_range)
%     alpha = para_range(i);
%     for j = 1:length(para_range)
%         beta = para_range(j);
%         for k = 1:length(para_range)
%             gamma = para_range(k);
%             [result,~,~,~,~] = CSTMSC_manifold(X,alpha,beta,gamma,gt);
%             result_final(i,j,k) = result(1);
%         end
%     end
% end
%% 运行10次
% alpha = 1e-3; 
% beta = 1e-1;
% gamma = 1e2;
% result = zeros(10,7);
% for i = 1:10
%     [result(i,:),~,~,~,~] = CSTMSC_manifold(X,alpha,beta,gamma,gt);
% end
% result_mean = mean(result);
% result_std = std(result);
%% S1和S2
% Cluster_num = size(unique(gt),1);
% s1 = zeros(2000);s2 = zeros(2000);
% for i = 1:2
%     s1 = s1 + S1{i};
%     s2 = s2 + S2{i};
% end
% s1 = s1 ./ 2;
% s2 = s2 ./ 2;
% y1 = spectral_clustering(s1,Cluster_num);
% result_s1 = Clustering8Measure(gt, y1');
% y2 = spectral_clustering(s2,Cluster_num);
% result_s2 = Clustering8Measure(gt, y2');