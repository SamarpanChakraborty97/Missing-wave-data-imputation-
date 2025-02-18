clear all
close all
clc

src='067p1_d26.nc';

ncdisp(src)

info=ncinfo(src,'xyzZDisplacement');
Starttime=ncread(src,'xyzStartTime');
samplerate=ncread(src,'xyzSampleRate');
filt_delay=ncread(src,'xyzFilterDelay');
BeginDate=datetime(1970,1,1)+seconds(Starttime-filt_delay)
EndDate=datetime(1970,1,1)+seconds(Starttime-filt_delay)+seconds(info.Size/samplerate)

eps=1*10^-3; % Set the parameter epsilon (cf. equation (1) and (2))
alpha = 0.8;
N_pts_per_wave=1;
N_windows=100; % Number of windows for comparison
N_MC=1000; % Number of Monte Carlo samples while solving the nonlinear function optimization for frequencies (4)
N_fmodes=16; % Number of elementary waves/ Fourier modes, i.e. parameter Nf in equation % (1)

% Number of waves in the inital fitting interval. For the given
% observations 55 wave corresponds to approximately 15 min of observations.

N_waves=20;

window_size=20*10^3;

h_tmp=cell(2,1);
t_tmp=cell(2,1);
hz_slowflow=cell(2,1);
slow_vars=cell(2,1);

% select a random start time
w_start=randi(info.Size-window_size);

hz=ncread(src,'xyzZDisplacement',w_start,window_size).';

% extract extreme values, i.e. trough depths and crests heights
[tmp_wave_height ,tmp_wave_idx,zero_idx, ~]=my_wave_height_filter(hz ,N_pts_per_wave);

ts=[tmp_wave_idx];%  tmp_t0];%[1:window_size];%
hs=[tmp_wave_height];% zeros(1,length(zero_idx))];%hz(jj,:);%

[ts,id]=sort(ts);

hs=hs(id);
missing_len = ceil(samplerate * 60); %1 minute of missing data

t1=ts(1:N_waves(1));
h1=hs(1:N_waves(1));

t2=ts(N_waves(1)+missing_len:N_waves(1)+missing_len+N_waves(1));
h2=hs(N_waves(1)+missing_len:N_waves(1)+missing_len+N_waves(1));

t_plot = (1:t1(end));
h_plot = hz(1:t1(end));

t2_plot = (t2(1):t2(end));
h2_plot = hz(t2(1):t2(end));

%%%%% Before the missing data %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if N_fmodes<9
    % If less than nine Fourier modes are selected a MATLAB routine can be
    % used for fitting.
    str_fmode=append('sin',num2str(N_fmodes));
    my_fit=fit(t1.',(h1-mean(h1)).',str_fmode);
    err_const=sum((h1-mean(h1)-my_fit(t1).').^2);
    coeff2=coeffvalues(my_fit);
    ws=sort(coeff2(2:3:end));
    tic
    % Monte Carlo sampling for optimization (4)
    [ws_MC, ~]= my_Freq_MC_fit(t1,h1,err_const,N_fmodes,ws,N_MC,samplerate);
    toc
else
    ws=zeros(1,N_fmodes);
    err_const=sum((h1-mean(h1)).^2);
    tic
    % Monte Carlo sampling for optimization (4)
    [ws_MC, ~]= my_Freq_MC_fit2(t1,h1,err_const,N_fmodes,ws,N_MC,samplerate);
    toc
end
% Fitting the slowly varying amplitudes, i.e. solving the linear
% optimization (5), for the first three minutes (cf. Fig. 3).
tic
[hz_slowflow{1} ,slow_vars{1}, ~]= my_SF_fit(t1, h1,ws_MC,eps,alpha);
    
toc
h_tmp{1}=h1;
t_tmp{1}=t1;

slow_vars_fit=slow_vars{1};
tt=t_tmp{1}(1):t_tmp{1}(end);
    
% Interpolation yielding uniform time series for the slowly varying amplitudes. 
slow_vars_interp1 = interp1(t_tmp{1},slow_vars_fit.',tt,'spline','extrap').';

%m0= mean(slow_vars_interp,2);
%m1=std(slow_vars_interp,0,2);
%slow_vars_nomean=(slow_vars_interp-m0)./m1;

writematrix(slow_vars_interp1,'Slow_amp_prev.csv','Delimiter',',');
hz_slowflow_full_truth=SF2fulltime(t_tmp{1},t_tmp{1},slow_vars{1},ws_MC,N_fmodes);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% After the missing data %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if N_fmodes<9
    % If less than nine Fourier modes are selected a MATLAB rutine can be
    % used for fitting.
    str_fmode=append('sin',num2str(N_fmodes));
    my_fit=fit(t2.',(h2-mean(h2)).',str_fmode);
    err_const=sum((h2-mean(h2)-my_fit(t2).').^2);
    coeff2=coeffvalues(my_fit);
    ws=sort(coeff2(2:3:end));
    tic
    % Monte Carlo sampling for optimization (4)
    [ws_MC, err_MC]= my_Freq_MC_fit(t2,h2,err_const,N_fmodes,ws,N_MC,samplerate);
    toc
else
    ws=zeros(1,N_fmodes);
    err_const=sum((h2-mean(h2)).^2);
    tic
    % Monte Carlo sampling for optimization (4)
    [ws_MC, err_MC]= my_Freq_MC_fit2(t2,h2,err_const,N_fmodes,ws,N_MC,samplerate);
    toc
end
% Fitting the slowly varying amplitudes, i.e. solving the linear
% optimization (5), for the first three minutes (cf. Fig. 3).
tic
[hz_slowflow{2} ,slow_vars{2}, ~]= my_SF_fit(t2, h2,ws_MC,eps,alpha);
    
toc
h_tmp{2}=h2;
t_tmp{2}=t2;

slow_vars_fit=slow_vars{2};
tt=t_tmp{2}(1):t_tmp{2}(end);
    
% Interpolation yielding uniform time series for the slowly varying amplitudes. 
slow_vars_interp2 = interp1(t_tmp{2},slow_vars_fit.',tt,'spline','extrap').';

%m0= mean(slow_vars_interp,2);
%m1=std(slow_vars_interp,0,2);
%slow_vars_nomean2=(slow_vars_interp-m0)./m1;

writematrix(slow_vars_interp2,'Slow_amp_follow.csv','Delimiter',',');
hz_slowflow_full_truth2=SF2fulltime(t_tmp{2},t_tmp{2},slow_vars{2},ws_MC,N_fmodes);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
f=gcf;
plot(t1, hz_slowflow_full_truth,'k-.','Linewidth',2)
hold on;
plot(t_plot, h_plot,'r-.','Linewidth',1.5)
saveas(f,'prevFit1.tiff')

figure;
f=gcf;
plot(t2, hz_slowflow_full_truth2,'k-.','Linewidth',2)
hold on;
plot(t2_plot, h2_plot,'r-.','Linewidth',1.5)
saveas(f,'followFit1.tiff')

figure;
f=gcf;
for i=1:length(slow_vars_interp1(:,1))
    plot(slow_vars_interp1(i,:))
    hold on;
end
saveas(f,'prevAmps1.tiff')

figure;
f=gcf;
for i=1:length(slow_vars_interp2(:,2))
    plot(slow_vars_interp2(i,:))
    hold on;
end
saveas(f,'followAmps1.tiff')
    