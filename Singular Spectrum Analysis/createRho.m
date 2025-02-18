function rho_jk = createRho(M, N1, Y_miss)
D = zeros(N1,M);
for i=1:N1
    D(i,:) = Y_miss(i:i+M-1);
end

C = (1/N1)*(D'*D);
[~, vals, left_vecs] = eig(C);

[~,ind] = sort(diag(vals));
ind_sort = flip(ind,1);

rho_jk = left_vecs(:,ind_sort);
end
