clc
clear all

for p=1:10
    name1 = sprintf('EleFit_whole_%d.csv',p);
    Whole = readmatrix(name1);
    name2 = sprintf('EleFit_pre_%d.csv',p);
    Pre = readmatrix(name2);
    name3 = sprintf('EleFit_post_%d.csv',p);
    Post = readmatrix(name3);

    pre_length = length(Pre);
    post_length = length(Post);
    whole_length = length(Whole);

    miss_length = whole_length - (pre_length + post_length);

    y = Whole;
    mean_y = mean(y);
    std_y = std(y);
    y_tot = (y - mean_y) ./ std_y;

    start_i = pre_length+1;
    end_i = pre_length+miss_length;
    Min = 5;

    Y_miss = y_tot;
    Y_miss(start_i:end_i)= 0;

    M = 400;
    N = length(Y_miss);
    N1 = N-M+1;

    D = zeros(N1,M);
    for i=1:N1
        D(i,:) = Y_miss(i:i+M-1);
    end

    C = (1/N1)*(D'*D);
    [right_vecs, vals, left_vecs] = eig(C);

    [d,ind] = sort(diag(vals));
    Eigvals = flip(d,1);
    ind_sort = flip(ind,1);

    rho_jk = left_vecs(:,ind_sort);

    K = 30;

    rec_current = Y_miss;

    t_1 = tic;
    k=1;
    while k<=K
        convergence = 15;
        while convergence >= 1e-5
            rho_jk = createRho(M, N1, rec_current);
            Ak = createEOFs(N1, M, rec_current, rho_jk, k);
            rec_current = recreateData(M, N, N1, k, Ak, rho_jk, y_tot, start_i, end_i);    
        
            rec_last = rec_current;
        
            rho_jk = createRho(M, N1, rec_current);
            Ak = createEOFs(N1, M, rec_current, rho_jk, k);      
            rec_current = recreateData(M, N, N1, k, Ak, rho_jk, y_tot, start_i, end_i);
               
            convergence = mean(norm(rec_current(start_i:end_i) - rec_last(start_i:end_i)));
        end
        k = k+1;
    end
    time = toc(t_1);
    fprintf('Elapsed time is %5.5f seconds\n',time);
    
    rec_current = rec_current .* std_y + mean_y;

    name = sprintf('SSA_preds_%d.out',p);
    writematrix(rec_current,name,'Delimiter','tab','FileType','text');
end