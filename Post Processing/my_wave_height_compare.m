function [h_cmp, t_cmp]=my_wave_height_compare(true_wave_height,true_wave_time,wave_height,wave_time)
% Comparing the true and predicted wave heights
% The wave height closest to the actual wave height and within 10 sec (~ 13
% samples) is considered as the forecasted value. 

h_cmp=zeros(1,length(true_wave_height));
t_cmp=zeros(1,length(true_wave_height));

for iter_waves=1:length(true_wave_time)
    
    h_tmp=wave_height(abs(wave_time-true_wave_time(iter_waves))<13);
    t_tmp=wave_time(abs(wave_time-true_wave_time(iter_waves))<13);
    if isempty(h_tmp)==true
        h_cmp(iter_waves)=NaN;
        t_cmp(iter_waves)=NaN;
    else
        [~,id]=min(abs(h_tmp-true_wave_height(iter_waves)));
        h_cmp(iter_waves)=h_tmp(id(1));
        t_cmp(iter_waves)=t_tmp(id(1));
    end
        
end
end