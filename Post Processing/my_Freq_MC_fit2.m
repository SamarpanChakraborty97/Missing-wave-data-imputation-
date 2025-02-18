function [ws_MC, err_sv]= my_Freq_MC_fit2(ts,hs,err_const,N_fmodes,w0,N_MC,samplerate)


% From Hasselman et al. and buoy data freq sepctrum
f_low=0.02;
f_high=0.6/3;
% Start parpool for paralleilzation. 
if isempty(gcp('nocreate'))==1
    parpool
end
% Minimal frequency
w_low_per_sampl=f_low*2*pi*samplerate;
% Maximal frequncy
w_high_per_sampl=f_high*2*pi*samplerate;

% Initialize error and frequencies
err_tmp=zeros(N_MC,1);
ws_tmp=zeros(N_MC+1,N_fmodes);
err_tmp(end)=err_const;
ws_tmp(end,:)=w0;

% build cost function
my_cost_fcn1=@(pars) 0;
% For each N_fmodes the cost function is extended by a sin and cos function
for iter_fmodes=1:N_fmodes
    idx1=3*(iter_fmodes-1);
    my_cost_fcn1=@(pars) my_cost_fcn1(pars(1:idx1))+pars(idx1+1).*cos(pars(idx1+3).*ts)+pars(idx1+2).*sin(pars(idx1+3).*ts);
    
    %my_cost_fcn2=@(pars) my_cost_fcn2(pars(1:idx1))-pars(idx1+1)*pars(idx1+3).*sin(pars(idx1+3).*ts)+pars(idx1+2)*pars(idx1+3).*cos(pars(idx1+3).*ts);
    
end
% The final cost function is to minimize the error
my_cost_fcn=@(pars)  my_cost_fcn1(pars)-hs;% my_cost_fcn2(pars)];
opts = optimoptions('lsqnonlin','MaxFunctionEvaluations',10^5,'FunctionTolerance',10^-6,'MaxIterations',10^4,'Display','off');%


%ws=w_low_per_sampl+(w_high_per_sampl-w_low_per_sampl).*rand(N_MC,N_fmodes);
parfor iter_MC=1:N_MC
    ws_start=w_low_per_sampl+(w_high_per_sampl-w_low_per_sampl).*rand(1,N_fmodes);%[0.1 0.2 0.3 0.4];
    pars0=zeros(3*N_fmodes,1);
    pars0(3:3:end)=ws_start;
    [coeffs, err_tmp(iter_MC)]=lsqnonlin(my_cost_fcn,pars0,[],[],opts)
  
    
    ws_tmp(iter_MC,:)=sort(coeffs(3:3:end));
      
end

[~, idx]=min(err_tmp);
err_sv=err_tmp(idx);

ws_MC=ws_tmp(idx,:);

end