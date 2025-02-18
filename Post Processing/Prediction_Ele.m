clear all
close all
clc

src='071p1_d19.nc';

ncdisp(src)

info=ncinfo(src,'xyzZDisplacement');
Starttime=ncread(src,'xyzStartTime');
samplerate=ncread(src,'xyzSampleRate');
filt_delay=ncread(src,'xyzFilterDelay');
BeginDate=datetime(1970,1,1)+seconds(Starttime-filt_delay);
EndDate=datetime(1970,1,1)+seconds(Starttime-filt_delay)+seconds(info.Size/samplerate);

eps=1*10^-3; % Set the parameter epsilon (cf. equation (1) and (2))
alpha = 1*10^-3;
N_pts_per_wave=1;
N_windows=100; % Number of windows for comparison
N_MC=1000; % Number of Monte Carlo samples while solving the nonlinear function optimization for frequencies (4)
N_fmodes=16; % Number of elementary waves/ Fourier modes, i.e. parameter Nf in equation % (1)

% Number of waves in the inital fitting interval. For the given
% observations 55 wave corresponds to approximately 15 min of observations.

num_minutes = 10;
num_miss_minutes = 1;
num_total_minutes = (num_minutes)*2 + num_miss_minutes;

window_size_pre = ceil(num_minutes * 60 * samplerate);
window_size_post = ceil(num_minutes * 60 * samplerate);
window_size_miss = ceil(num_miss_minutes * 60 * samplerate);
window_size_tot = window_size_pre + window_size_post + window_size_miss;

initial_window_start = 10000;
Num_Windows = 1;
Max_Num_Windows = 100;

% select a random start time
% Truth_Miss = zeros(100,80);

truth_cum = [];
lstm_cum = [];
cnn_lstm_cum = [];
ssa_cum = [];
fit_cum = [];
baseline_cum = [];

mins = 1;

LSTM = zeros(Num_Windows,3);        t_LSTM = zeros(Num_Windows,2);          Max_LSTM = zeros(Num_Windows,1);
SSA = zeros(Num_Windows,3);         t_SSA = zeros(Num_Windows,2);           Max_SSA = zeros(Num_Windows,1);
CNN_LSTM = zeros(Num_Windows,3);    t_CNN_LSTM = zeros(Num_Windows,2);      Max_CNN_LSTM = zeros(Num_Windows,1);
Fit = zeros(Num_Windows,3);         t_Fit = zeros(Num_Windows,2);           Max_Fit = zeros(Num_Windows,1);
Baseline = zeros(Num_Windows,3);    t_Baseline = zeros(Num_Windows,2);      Max_Baseline = zeros(Num_Windows,1);

w_start = initial_window_start : 18960 : (18960*Max_Num_Windows)+initial_window_start;
rand_sample = randi([1 100])
for i=rand_sample:rand_sample
    
    %h_tmp=cell(3, 1);
    %t_tmp=cell(3, 1);
    %hz_slowflow=cell(3, 1);
    %slow_vars=cell(3, 1);

    hz=ncread(src,'xyzZDisplacement',w_start(i),window_size_tot).';
    
    % extract extreme values, i.e. trough depths and crests heights
    [tmp_wave_height ,tmp_wave_idx,zero_idx, ~]=my_wave_height_filter(hz ,N_pts_per_wave);
    
    ts=[tmp_wave_idx];%  tmp_t0];%[1:window_size];%
    hs=[tmp_wave_height];% zeros(1,length(zero_idx))];%hz(jj,:);%
    
    [ts,id]=sort(ts);
    hs=hs(id);
    
    N_waves_tot = length(hs);
    N_waves_miss = 10 * num_miss_minutes;
    N_waves_pre = ceil((length(hs)-N_waves_miss)/2);
    N_waves_post = N_waves_tot - N_waves_miss - N_waves_pre;
    
    t_full=ts(1:N_waves_tot);
    h_full=hs(1:N_waves_tot);
    
    t_pre=ts(1:N_waves_pre);
    h_pre=hs(1:N_waves_pre);
    
    t_miss=ts(N_waves_pre:N_waves_pre+N_waves_miss);
    h_miss=hs(N_waves_pre:N_waves_pre+N_waves_miss);
    
    t_post=ts(N_waves_pre+N_waves_miss:N_waves_tot);
    h_post=hs(N_waves_pre+N_waves_miss:N_waves_tot);
    
    %t_full_plot = (t_full(1):t_full(end));
    %h_full_plot = hz(t_full(1):t_full(end));
    
    %t_pre_plot = (t_pre(1):t_pre(end));
    %h_pre_plot = hz(t_pre(1):t_pre(end));
    
    %t_post_plot = (t_post(1):t_post(end));
    %h_post_plot = hz(t_post(1):t_post(end));
    
    t_miss_plot = (t_miss(1)+1:t_miss(end)-1);
    h_miss_plot = hz(t_miss(1)+1:t_miss(end)-1);
    
    %Truth_Miss(i,1:length(t_miss_plot)) = h_miss_plot;  % Truth
    
    name = sprintf('Omegas_%i.csv',i);
    freqs = readmatrix(name);               % Frequencies
    
    %%%%%% FULL SURFACE ELEVATION TIME SERIES %%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%% FULL SURFACE ELEVATION TIME SERIES %%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%% LSTM %%%%%%%   
    name = sprintf('Preds_lstm_%i.out',i);
    lstm = readmatrix(name,'FileType','text');
    hz_slowflow_full_lstm=SF2fulltime(t_miss_plot,t_miss_plot,lstm,freqs,N_fmodes);
    %%%%%%%%%%%%%%%%%%%
    
    %%%%% SINGULAR SPECTRUM ANALYSIS METHOD %%%%%
    name = sprintf('SSA_preds_%i.out',i);
    ssa_full = dlmread(name);
    ssa = ssa_full(t_miss(1)+1:t_miss(end)-1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%% CNN+LSTM_FIT %%%%%%%   
    name = sprintf('Preds_cnn+lstm_%i.out',i);
    cnn_lstm_fit = dlmread(name);
    %%%%%%%%%%%%%%%%%%%
    
    %%%%%% CNN+LSTM_AMPLITUDES %%%%%%%   
    name = sprintf('Preds_pre_cnn+lstm_%i.out',i);
    cnn_lstm_amps = dlmread(name);
    hz_slowflow_full_cnn_lstm=SF2fulltime(t_miss_plot,t_miss_plot,cnn_lstm_amps,freqs,N_fmodes);
    %%%%%%%%%%%%%%%%%%%
    
    %%%%% FIT %%%%%%
    name = sprintf('Slow_amp_whole_%i.csv',i);
    fit_full = readmatrix(name);
    fit = fit_full(:,t_miss(1)+1:t_miss(end)-1);
    hz_slowflow_full_fit=SF2fulltime(t_miss_plot,t_miss_plot,fit,freqs,N_fmodes);
    %%%%%%%%%%%%%%%%
    
    %%%%% BASELINE %%%%%
    % Half of the mean wave period
    T_mean=mean(diff(t_pre));
    % Number of waves in the forecast
    N_waves_forecast=N_waves_miss;
    % Mean wave height
    mean_wave_guess=repmat(mean(abs(h_pre)),1,N_waves_forecast);
    %mean_wave_sign=ones(1,N_waves_forecast);
    % If first value is positive the benchmark starts with a positive guess
    % and negtive if otherwise. The other values alternate. ( i.e. crest,
    % through, crest, trough,... or trough, crest,trough, crest,...)
    if h_miss(1)<0
        mean_wave_guess(2:2:end)=-mean_wave_guess(2:2:end);
    else
        mean_wave_guess(1:2:end)=-mean_wave_guess(1:2:end);
    end
    
%     mean_wave_guess
%     T_mean.* (0:N_waves_forecast-1)
    
%     name = sprintf('Slow_amp_pre_%i.csv',i);
%     fit_pre = readmatrix(name);
%     baseline = zeros(33, length(t_miss_plot));
%     for j=1:33
% 	    baseline(j,:) =  repmat(mean(fit_pre(j,:),2),1,length(t_miss_plot));
%     end
%     hz_slowflow_full_baseline=SF2fulltime(t_miss_plot,t_miss_plot,baseline,freqs,N_fmodes);
    %%%%%%%%%%%%%%%%%%%
    %%%%%%%%%% SURFACE ELEVATION COMPLETED %%%%%%%%%%
    
    %%%%%%%%% CRESTS AND PEAKS EXTRACTION %%%%%%%%%
    [true_height, true_time,~,~]=my_wave_height_filter(h_miss_plot, N_pts_per_wave);
    [fit_wave_height ,fit_time,~, ~]=my_wave_height_filter(hz_slowflow_full_fit, N_pts_per_wave);
    [lstm_wave_height ,lstm_forecast_time,~, ~]=my_wave_height_filter(hz_slowflow_full_lstm, N_pts_per_wave);
    [cnn_lstm_fit_wave_height ,cnn_lstm_fit_forecast_time,~, ~]=my_wave_height_filter(cnn_lstm_fit, N_pts_per_wave);
    [cnn_lstm_amp_wave_height ,cnn_lstm_amp_forecast_time,~, ~]=my_wave_height_filter(hz_slowflow_full_cnn_lstm, N_pts_per_wave);
    [ssa_wave_height ,ssa_forecast_time,~, ~]=my_wave_height_filter(ssa, N_pts_per_wave);
    %[~,baseline_time,~, ~]=my_wave_height_filter(hz_slowflow_full_baseline, N_pts_per_wave);
    
    %true_time
    
    %length(baseline_wave_height)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%% COMPARING OBTAINED PEAKS WITH TRUTH %%%%%%%%
    [h_cmp_fit,t_cmp_fit] = my_wave_height_compare(true_height,true_time,fit_wave_height,fit_time);
    [h_cmp_lstm,t_cmp_lstm] = my_wave_height_compare(true_height,true_time,lstm_wave_height,lstm_forecast_time);
    [h_cmp_cnn_fit_lstm,t_cmp_cnn_fit_lstm] = my_wave_height_compare(true_height,true_time,cnn_lstm_fit_wave_height,cnn_lstm_fit_forecast_time);
    [h_cmp_cnn_amp_lstm,t_cmp_cnn_amp_lstm] = my_wave_height_compare(true_height,true_time,cnn_lstm_amp_wave_height,cnn_lstm_amp_forecast_time);
    [h_cmp_ssa, t_cmp_ssa] = my_wave_height_compare(true_height,true_time,ssa_wave_height,ssa_forecast_time);
    %[h_cmp_baseline, t_cmp_baseline] = my_wave_height_compare(true_height,true_time,baseline_wave_height,true_time);
    [h_cmp_baseline, t_cmp_baseline]=my_wave_height_compare(true_height,true_time,mean_wave_guess,T_mean.* (0:N_waves_forecast-1));
    [h_cmp_truth, t_cmp_truth] = my_wave_height_compare(true_height,true_time,true_height,true_time);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%% PEAKS TILL 1 MINUTE %%%%%%%%%
    idx_1min=length(t_cmp_truth);
    
    h_cmp_truth_1min = h_cmp_truth;                              % Truth 1 minute peaks
    t_cmp_truth_1min = t_cmp_truth;                              % Truth 1 minute peak times

    h_cmp_baseline_1min = h_cmp_baseline(1:idx_1min);            % Baseline 1 minute peaks
    t_cmp_baseline_1min = t_cmp_baseline(1:idx_1min);            % Baseline 1 minute peak times

    h_cmp_fit_1min = h_cmp_fit(1:idx_1min);                      % Fit 1 minute peaks
    t_cmp_fit_1min = t_cmp_fit(1:idx_1min);                      % Fit 1 minute peak times

    h_cmp_lstm_1min = h_cmp_lstm(1:idx_1min);                    % LSTM 1 minute peaks
    t_cmp_lstm_1min = t_cmp_lstm(1:idx_1min);                    % LSTM 1 minute peak times
    
    h_cmp_cnn_lstm_fit_1min = h_cmp_cnn_fit_lstm(1:idx_1min);    % CNN+LSTM 1 minute peaks
    t_cmp_cnn_lstm_fit_1min = t_cmp_cnn_fit_lstm(1:idx_1min);    % CNN+LSTM 1 minute peak times
    
    h_cmp_cnn_lstm_amp_1min = h_cmp_cnn_amp_lstm(1:idx_1min);    % CNN+LSTM SLOW 1 minute peaks
    t_cmp_cnn_lstm_amp_1min = t_cmp_cnn_amp_lstm(1:idx_1min);    % CNN+LSTM SLOW 1 minute peak times
    
    h_cmp_ssa_1min = h_cmp_ssa(1:idx_1min);                      % SSA 1 minute peaks
    t_cmp_ssa_1min = t_cmp_ssa(1:idx_1min);                      % SSA 1 minute peak times 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    figure;
    g = gcf;
    
      
    subplot(4,1,1);
    plot(h_cmp_truth_1min, 'k-.','Linewidth',2.0);
    hold on;
    grid on;
    grid minor;
    plot(h_cmp_lstm_1min, 'm-','Linewidth',2.0);
    legend({'Truth peaks','LSTM prediction peaks'},'FontName','Times','FontSize',15,'interpreter','latex','Location','best');
    set(gca,'fontsize', 21) 
    
    
    subplot(4,1,2);
    plot(h_cmp_truth_1min, 'k-.','Linewidth',2.0);
    hold on;
    grid on;
    grid minor;
    plot(h_cmp_cnn_lstm_fit_1min, 'b-','Linewidth',2.0);
    legend({'Truth peaks','CNN+LSTM prediction peaks'},'FontName','Times','FontSize',15,'interpreter','latex','Location','best');
    set(gca,'fontsize', 21) 
    
    
    subplot(4,1,3);
    plot(h_cmp_truth_1min, 'k-.','Linewidth',2.0);
    hold on;
    grid on;
    grid minor;
    plot(h_cmp_ssa_1min,'-','color',[0,0.5,0],'Linewidth',2.0);
    legend({'Truth peaks','SSA prediction peaks'},'FontName','Times','FontSize',15,'interpreter','latex','Location','best');
    set(gca,'fontsize', 21) 
    
    subplot(4,1,4);
    plot(h_cmp_truth_1min, 'k-.','Linewidth',2.0);
    hold on;
    grid on;
    grid minor;
    plot(h_cmp_fit_1min, 'r-','Linewidth',2.0);
    legend({'Truth peaks','Fit peaks'},'FontName','Times','FontSize',15,'interpreter','latex','Location','best');
    set(gca,'fontsize', 21) 
    
    name = sprintf('Peaks for %i minute for different models over a sample window',mins);
    sgtitle(name,'interpreter','latex','fontweight','normal','fontsize',24);

    g.PaperUnits = 'inches';
    g.PaperPosition = [0 0 15 20];
    name2 = sprintf('Sample comparisons (fit) plots for %i minute for different models over a sample window %i.tiff',mins, rand_sample);
    print(g,name2,'-dtiff','-r600');
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all