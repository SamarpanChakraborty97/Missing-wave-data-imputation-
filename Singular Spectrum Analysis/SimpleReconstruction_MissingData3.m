clc
clear all

a = 4;
lambda = 0.085;
omega = 2*pi;
f = omega / (2*pi);

TP = 50;
t = 0:1/(10*f):TP*(1/f);
y = a.*exp(-lambda*t).*sin(omega*t);

y1 = y;
Y_saw_profile = horzcat(y,y1,y1);

plot(Y_saw_profile)

delT = 1/(10*f);
T = 0: delT: 30*TP*delT + 2*delT;

Y_miss = Y_saw_profile;
Y_miss(505:605)= 0;

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

M = 45;
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
%plot(Eigvals,'r.','MarkerSize',16)

M = length(rho_jk(:,1));
% A1 = createEOFs(N1, M, Y_miss, rho_jk, 1);
% A2 = createEOFs(N1, M, Y_miss, rho_jk, 2);
% A3 = createEOFs(N1, M, Y_miss, rho_jk, 3);
% A4 = createEOFs(N1, M, Y_miss, rho_jk, 4);

%Reconstruction
% Rec1 = zeros(length(Y_saw_profile),1);
K = 10;

% Ak = zeros(K,N1);
% Ak(1,:) = A1;
% Ak(2,:) = A2;
% Ak(3,:) = A3;
% Ak(4,:) = A4;

rec_current = Y_miss;

% A2 = createEOFs(N1, M, rec_current, rho_jk, 2);

% plot(A1,'b')
% hold on;
% plot(A2,'r')

% figure
% rec_current1 = recreateData(M, N, N1, 1, A1, rho_jk, Y_saw_profile);
% rec_current2 = recreateData(M, N, N1, 2, A2, rho_jk, Y_saw_profile);
% plot(rec_current1,'b')
% hold on;
% plot(rec_current2,'r')

k=1;
while k<=K
    convergence = 15;
    while convergence >= 1e-12
        rho_jk = createRho(M, N1, rec_current);
        Ak = createEOFs(N1, M, rec_current, rho_jk, k);
        rec_current = recreateData(M, N, N1, k, Ak, rho_jk, Y_saw_profile);    
        
        rec_last = rec_current;
        
        rho_jk = createRho(M, N1, rec_current);
        Ak = createEOFs(N1, M, rec_current, rho_jk, k);      
        rec_current = recreateData(M, N, N1, k, Ak, rho_jk, Y_saw_profile);
               
        convergence = norm(rec_current(505:605) - rec_last(505:605));
    end
    k = k+1;
end

L = floor(2*length(rec_current)/3);
figure
f = gcf;
plot(rec_current(1:L),'r-','linewidth',2.0)
hold on;
plot(Y_saw_profile(1:L),'k-','linewidth',1.0)
grid('on');
xlabel('Time(in seconds)');
ylabel('Y')
saveas(f,'3_bursts.png')

% Rec1 = recreateData(M, N, 1, Ak, rho_jk);
% rho_jk = createRho(M, N1, Rec1);
% A1 = createEOFs(N1, M, Rec1, rho_jk, 1);
% Rec1 = recreateData(M, N, 1, Ak, rho_jk);

% for i=1:length(Rec1)
%     if i <= M-1
%         sum1 = 0;   
%         for k=1:2
%             for j=1:i
%                 ak = Ak(k,i-j+1);
%                 rho_k = rho_jk(j,k);
%                 sum1 = sum1 + ak * rho_k;
%             end
%         end
%         Mt = (1/(i+1));
%         Rec1(i) = Mt * sum1;
% 
%     elseif (M <= i)&& (i<= N1)
%         sum2 = 0;
%         for k=1:2
%             for j=1:M
%                 ak = Ak(k,i-j+1);
%                 rho_k = rho_jk(j,k);
%                 sum2 = sum2 + ak * rho_k;
%             end
%         end
%         Mt = (1/M);
%         Rec1(i) = Mt * sum2;
%         
%     elseif (N1+1 <= i)&&(i <=N)
%         sum3 = 0;
%         for k=1:2
%             for j=i-N+M:M
%                 ak = Ak(k,i-j+1);
%                 rho_k = rho_jk(j,k);
%                 sum3 = sum3 + ak * rho_k;
%             end
%         end
%         Mt = (1/(N-i+1));
%         Rec1(i) = Mt * sum3;
%     end
% end

% figure;
% plot(A1+A2+A3+A4,'k','linewidth',2.0)
% hold on;
% plot(Y_saw_profile,'b','linewidth',0.5)
% %plot(A3,'r','linewidth',0.5)
% %plot(A4,'g','linewidth',0.5)
% xlabel('Time(seconds)')
% ylabel('PC1')
% grid('on')

% f = figure;
% ax = gca;
% f.Position = [100 100 1500 300];
% plot(Rec1,'k-','linewidth',2.0)
% hold on;
% plot(Y_saw_profile,'b-','linewidth',2.0)
% grid('on');
% xlabel('Time(in seconds)');
% ylabel('Y')

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

function Y_reg = recreateData(M, N, N1, K, Ak, rho_jk, Y_saw_profile)
Y_reg = Y_saw_profile;
for i=505:605
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