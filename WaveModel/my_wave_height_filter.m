function [wave_height,wave_idx,zero_crossing,t0]=my_wave_height_filter(hz,N_pts_per_wave)

hz_mean=mean(hz);
zero_crossing=find(abs(diff(sign(hz-hz_mean) ))==2);

insig_waves =diff(zero_crossing)<N_pts_per_wave;

zero_crossing( insig_waves )=[];

wave_height=zeros(1,length(zero_crossing)-1);
wave_idx=zeros(1,length(zero_crossing)-1);

for ii=1:length(zero_crossing)-1
    [~, tmp_idx]=max(abs(hz((zero_crossing(ii)+1):(zero_crossing(ii+1)))));
    wave_idx(ii)=tmp_idx+zero_crossing(ii);
    wave_height(ii)= hz(wave_idx(ii));
    
end

idx=find(diff(sign(wave_height))==0,1);
while isempty(idx)~=1
    tmp_heights=[wave_height(idx) wave_height(idx+1)];
    tmp_wave_idxs=[wave_idx(idx) wave_idx(idx+1)];
    [~, tmp_idx]=max(abs( tmp_heights));
    wave_height(idx)=tmp_heights( tmp_idx);
    wave_idx(idx)=tmp_wave_idxs(tmp_idx);
    wave_height(idx+1)=[];
    wave_idx(idx+1)=[];
    zero_crossing(idx+1)=[];
    idx=find(diff(sign(wave_height))==0,1);
end

for ii=1:length(zero_crossing)
    % if zero_crossing(ii+1)-zero_crossing(ii)>N_pts_per_wave-1
    t0(ii)=zero_crossing(ii)-hz(zero_crossing(ii))/(hz(zero_crossing(ii)+1)-hz(zero_crossing(ii)));
end
end