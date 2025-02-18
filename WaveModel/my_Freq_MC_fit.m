function [ws_MC, err_sv]= my_Freq_MC_fit(ts,hs,err_const,N_fmodes,w0,N_MC,samplerate)
% For comments see my_Freq_MC_fit2
str_fmode=append('sin',num2str(N_fmodes));

% From Hasselman et al. and buoy data freq sepctrum
f_low=0.02;
f_high=0.6/3;
if isempty(gcp('nocreate'))==1
    parpool
end
w_low_per_sampl=f_low*2*pi*samplerate;
w_high_per_sampl=f_high*2*pi*samplerate;

err_tmp=zeros(N_MC,1);
ws_tmp=zeros(N_MC+1,N_fmodes);
err_tmp(end)=err_const;
ws_tmp(end,:)=w0;


parfor iter_MC=1:N_MC
    opts=fitoptions(str_fmode);
    opts.StartPoint=zeros(1,N_fmodes*3);
    
    opts.StartPoint(2:3:end)=w_low_per_sampl+(w_high_per_sampl-w_low_per_sampl).*rand(1,N_fmodes);%[0.1 0.2 0.3 0.4];
    my_fit=fit(ts.',(hs-mean(hs)).',str_fmode,opts);
    err_tmp(iter_MC)=sum((hs-mean(hs)-my_fit(ts).').^2);
    
    
    coeff2=coeffvalues(my_fit);
    ws_tmp(iter_MC,:)=sort(coeff2(2:3:end));
        
end

[~, idx]=min(err_tmp);
err_sv=err_tmp(idx);

ws_MC=ws_tmp(idx,:);

end