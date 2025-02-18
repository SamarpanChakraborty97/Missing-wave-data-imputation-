a = 4;
lambda = 0.085;
omega = 2*pi;
f = omega / (2*pi);

TP = 50;
t = 0:1/(10*f):TP*(1/f);
y = a.*exp(-lambda*t).*sin(omega*t);

y1 = y;
Y_saw_profile = horzcat(y,y1,y1,y1);
delT = 1/(10*f);
T = 0:delT:40*TP*delT + 3*delT;

plot(T, Y_saw_profile);

M = 5;
N = length(Y_saw_profile);
N1 = N-M+1;

D = zeros(N1,M);
for i=1:N1
    D(i,:) = Y_saw_profile(i:i+M-1);
end

C = (1/N1)*(D'*D);
[right_vecs, vals, left_vecs] = eig(C);

[d,ind] = sort(diag(vals));
Eigvals = flip(d,1);
ind_sort = flip(ind,1);

rho_jk = left_vecs(:,ind_sort);
plot(Eigvals,'r.','MarkerSize',16)

M = length(rho_jk(:,1));

A1 = zeros(N1,1);
A2 = zeros(N1,1);
A3 = zeros(N1,1);
A4 = zeros(N1,1);

for i=1:N1
    sum1 = 0;
    sum2 = 0;
    sum3 = 0;
    sum4 = 0;
    
    for j=1:M
        rho = rho_jk(j,1);
        y = Y_saw_profile(i+j-1);
        sum1 = sum1 + y * rho;     
    end
    A1(i) = sum1;

    for j=1:M
        rho = rho_jk(j,2);
        y = Y_saw_profile(i+j-1);
        sum2 = sum2 + y * rho;              
    end
    A2(i) = sum2;   
    
    for j=1:M
        rho = rho_jk(j,3);
        y = Y_saw_profile(i+j-1);
        sum3 = sum3 + y * rho;                
    end
    A3(i) = sum3; 
    
    for j=1:M
        rho = rho_jk(j,4);
        y = Y_saw_profile(i+j-1);
        sum4 = sum4 + y * rho;        
    end
    A4(i) = sum4;

end

%Reconstruction
Rec1 = zeros(length(Y_saw_profile),1);
K = 4;

a = zeros(K,N1);
a(1,:) = A1;
a(2,:) = A2;
a(3,:) = A3;
a(4,:) = A4;

for i=1:length(Rec1)
    if i <= M-1
        sum1 = 0;   
        for k=1:K
            for j=1:i
                ak = a(k,i-j+1);
                rho_k = rho_jk(j,k);
                sum1 = sum1 + ak * rho_k;
            end
        end
        Mt = (1/(i+1));
        Rec1(i) = Mt * sum1;

    elseif (M <= i)&& (i<= N1)
        sum2 = 0;
        for k=1:K
            for j=1:M
                ak = a(k,i-j+1);
                rho_k = rho_jk(j,k);
                sum2 = sum2 + ak * rho_k;
            end
        end
        Mt = (1/M);
        Rec1(i) = Mt * sum2;
        
    elseif (N1+1 <= i)&&(i <=N)
        sum3 = 0;
        for k=1:K
            for j=i-N+M:M
                ak = a(k,i-j+1);
                rho_k = rho_jk(j,k);
                sum3 = sum3 + ak * rho_k;
            end
        end
        Mt = (1/(N-i+1));
        Rec1(i) = Mt * sum3;
    end
end

% figure;
% plot(A1+A2+A3+A4,'k','linewidth',2.0)
% hold on;
% plot(Y_saw_profile,'b','linewidth',0.5)
% %plot(A3,'r','linewidth',0.5)
% %plot(A4,'g','linewidth',0.5)
% xlabel('Time(seconds)')
% ylabel('PC1')
% grid('on')

figure;
plot(Rec1,'k-','linewidth',1.0)
hold on;
plot(Y_saw_profile,'b','linewidth',0.5)
%plot(A3,'r','linewidth',0.5)
%plot(A4,'g','linewidth',0.5)
xlabel('Time(seconds)')
ylabel('PC1')
grid('on')