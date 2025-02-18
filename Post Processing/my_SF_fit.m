function [hz_slowflow, slow_vars,err]= my_SF_fit(ts, hs,ws,eps,alpha)
% Fitting the slowly varing amplitdues

% Initializing variables
N_t=length(ts);
N_fmodes=length(ws);
M1=zeros(N_t,(2*N_fmodes+1)*N_t);
M2=zeros((2*N_fmodes+1)*N_t,(2*N_fmodes+1)*N_t);
M3=zeros(N_t,(2*N_fmodes+1)*N_t);

% Building matrices
for tt=1:N_t
    idx0=1+(tt-1)*(2*N_fmodes+1);
    idx1=idx0+N_fmodes-1;
    idx2=idx1+N_fmodes;
    % Matrix for the first summand in the cost function (2)
    M1(tt,idx0)=1;
    M1(tt,idx0+1:idx1+1)= cos(ws.*ts(tt));
    M1(tt,idx1+2:idx2+1)= sin(ws.*ts(tt));
    % Matrix for the third summand in the cost function (2)
    M3(tt,idx0+1:idx1+1)= -ws.*sin(ws.*ts(tt));
    M3(tt,idx1+2:idx2+1)= ws.*cos(ws.*ts(tt));
    % Matrix for the second summand in the cost function (2)
    % Finite difference appoximation of the derivative
    if tt<N_t
        del_t=(ts(tt+1)-ts(tt));
        idx1=1+(tt-1)*(2*N_fmodes+1);
        M2(idx1,idx1)=-1/del_t;
        M2(idx1,idx1+2*N_fmodes+1)=1/del_t;
        %M2(idx1,idx1-2*N_fmodes-1)=1/del_t;
        for iter_fmodes=1:N_fmodes
            M2(idx1+iter_fmodes,idx1+iter_fmodes)=-1/del_t;
            M2(idx1+iter_fmodes,idx1+iter_fmodes+2*N_fmodes+1)=1/del_t;
            M2(idx1+iter_fmodes+N_fmodes,idx1+iter_fmodes+N_fmodes)=-1/del_t;
            M2(idx1+iter_fmodes+N_fmodes,idx1+iter_fmodes+3*N_fmodes+1)=1/del_t;
        end
    else
        del_t=(ts(tt)-ts(tt-1));
        idx1=1+(N_t-1)*(2*N_fmodes+1);
        %         M2(idx1,idx1)=-1/del_t;
        %         M2(idx1,1)=1/del_t;
        
        M2(idx1,idx1)=1/del_t;
        M2(idx1,idx1-(2*N_fmodes+1))=M2(idx1,idx1-(2*N_fmodes+1))-1/del_t;
        
        for iter_fmodes=1:N_fmodes
            
            
         M2(idx1+iter_fmodes,idx1+iter_fmodes)= 1/del_t;
            M2(idx1+iter_fmodes,idx1+iter_fmodes-(2*N_fmodes+1))= M2(idx1+iter_fmodes,idx1+iter_fmodes-(2*N_fmodes+1))-1/del_t;
            
            M2(idx1+iter_fmodes+N_fmodes,idx1+iter_fmodes+N_fmodes )=1/del_t;
            M2(idx1+iter_fmodes+N_fmodes,idx1+iter_fmodes-N_fmodes-1)=M2(idx1+iter_fmodes+N_fmodes,idx1+iter_fmodes-N_fmodes-1)-1/del_t;
            
            
        end
    end
end
% Adding the matirces with the scales, alpha=0.5 and epsilon
Mbig=(M1.'*M1+1/eps.*M2.'*M2+  alpha.* M3.'*M3);
% Obtaining least square solution
Vs=Mbig\(M1.'*hs.');

% Writing result into the variable slow_flow
hz_slowflow=zeros(1,N_t);
slow_vars=zeros(2*N_fmodes,N_t);
for tt=1:N_t
    idx1=1+(tt-1)*(2*N_fmodes+1);
    
    hz_slowflow(tt)=Vs(idx1);
    slow_vars(1,tt)=Vs(idx1);
    for iter_fmodes=1:N_fmodes
        hz_slowflow(tt)=hz_slowflow(tt)+Vs(idx1+iter_fmodes)*cos(ws(iter_fmodes)*ts(tt))...
            + Vs(idx1+iter_fmodes+N_fmodes)*sin(ws(iter_fmodes)*ts(tt));
        slow_vars(1+iter_fmodes,tt)=Vs(idx1+iter_fmodes);
        slow_vars(1+N_fmodes+iter_fmodes,tt)=Vs(idx1+N_fmodes+iter_fmodes);
    end
end


err=sum((hz_slowflow-hs).^2);


end