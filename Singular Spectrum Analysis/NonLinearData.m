clc
clear all

omega1 = 2*pi/200;
omega2 = 2*pi/40;
omega3 = 2*pi/120;

t = 0:0.1:600;

y = sin(omega1 * t) .* cos((omega2 * t) + (pi/2)*sin(omega3 * t));

% plot(t,y)

start_i = 1050;
end_i = 1400;

Y_miss = y;
Y_miss(start_i:end_i)= 0;

% f = figure;
% ax = gca;
% f.Position = [100 100 1500 300];
% plot(T(1:504), Y_miss(1:504),'r-','linewidth',2);
% hold on;
% plot(T(606:end), Y_miss(606:end),'r-','linewidth',2);
% plot(T(505:605), Y_saw_profile(505:605),'k-','linewidth',2);
% grid('on');
% xlim([0 T(end)]);
% xlabel('Time(in seconds)');
% ylabel('Y')

M = 2000;
N = length(Y_miss);
N1 = N-M+1;

plot(Y_miss)
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
plot(Eigvals,'r.','MarkerSize',16)

K = 9;

rec_current = Y_miss;

k=1;
while k<=K
    convergence = 15;
    while convergence >= 1e-4
        rho_jk = createRho(M, N1, rec_current);
        Ak = createEOFs(N1, M, rec_current, rho_jk, k);
        rec_current = recreateData(M, N, N1, k, Ak, rho_jk, y, start_i, end_i);    
        
        rec_last = rec_current;
        
        rho_jk = createRho(M, N1, rec_current);
        Ak = createEOFs(N1, M, rec_current, rho_jk, k);      
        rec_current = recreateData(M, N, N1, k, Ak, rho_jk, y, start_i, end_i);
               
        convergence = norm(rec_current(start_i:end_i) - rec_last(start_i:end_i));
    end
    k = k+1;
    k
end

L = floor(2*length(rec_current)/3);
figure
f = gcf;
plot(t(1:start_i-1),rec_current(1:start_i-1),'r-','linewidth',2.0)
hold on;
plot(t(end_i+1:end),rec_current(end_i+1:end),'r-','linewidth',2.0)
plot(t(start_i:end_i),rec_current(start_i:end_i),'b-','linewidth',2.0)
plot(t,y(1:end),'k-','linewidth',1.0)
grid('on');
xlabel('Time(in seconds)');
ylabel('Y')
saveas(f,'3_bursts.png')

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