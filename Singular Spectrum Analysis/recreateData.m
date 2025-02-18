function Y_reg = recreateData(M, N, N1, K, Ak, rho_jk, Y_saw_profile, start_i, end_i)
Y_reg = Y_saw_profile;
for i=start_i:end_i
    if i <= M-1
        sum1 = 0;   
        for k=1:K
            for j=1:i
                ak = Ak(k,i-j+1);
                rho_k = rho_jk(j,k);
                sum1 = sum1 + ak * rho_k;
            end
        end
        Mt = (1/(i+1));
        Y_reg(i) = Mt * sum1;

    elseif (M <= i)&& (i<= N1)
        sum2 = 0;
        for k=1:K
            for j=1:M
                ak = Ak(k,i-j+1);
                rho_k = rho_jk(j,k);
                sum2 = sum2 + ak * rho_k;
            end
        end
        Mt = (1/M);
        Y_reg(i) = Mt * sum2;
        
    elseif (N1+1 <= i)&&(i <=N)
        sum3 = 0;
        for k=1:K
            for j=i-N+M:M
                ak = Ak(k,i-j+1);
                rho_k = rho_jk(j,k);
                sum3 = sum3 + ak * rho_k;
            end
        end
        Mt = (1/(N-i+1));
        Y_reg(i) = Mt * sum3;
    end
end
end
