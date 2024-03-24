function [d] = stiefel_distance(X)
%STIEFEL_DISTANCE 计算Stiefel流形地线距离，stiefel_d(X,Y)=dim-tr(X^T*Y)
N=size(X,2);
V=size(X,1);
d=abs(V-X'*X);
end

