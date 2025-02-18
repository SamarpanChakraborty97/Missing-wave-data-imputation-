function a = createEOFs(N1, M, Y, rho_jk, K)
a = zeros(K,N1);
for k=1:K
    for i=1:N1
        sum1 = 0;
        for j=1:M
            rho = rho_jk(j,k);
            y = Y(i+j-1);
            sum1 = sum1 + y * rho;
        end
        a(k,i) = sum1;
    end
end
end
