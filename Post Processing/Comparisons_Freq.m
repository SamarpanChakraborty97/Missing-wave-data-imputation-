clear all
close all
clc

src='067p1_d26.nc';

ncdisp(src)

info=ncinfo(src,'xyzZDisplacement');
Starttime=ncread(src,'xyzStartTime');
samplerate=ncread(src,'xyzSampleRate');
filt_delay=ncread(src,'xyzFilterDelay');
BeginDate=datetime(1970,1,1)+seconds(Starttime-filt_delay);
EndDate=datetime(1970,1,1)+seconds(Starttime-filt_delay)+seconds(info.Size/samplerate);

eps=1*10^-3; % Set the parameter epsilon (cf. equation (1) and (2))
alpha = 0.5;
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

Num_Windows = 100;
initial_window_start = 5;

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

w_start = initial_window_start : 500 : (500*Num_Windows)+initial_window_start;

min_peaks = 500;
max_peaks = -2;

for i=1:Num_Windows
    
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
    
    %%%%%% LSTM %%%%%%%   
    name = sprintf('Preds_pre_lstm_%i.out',i);
    lstm_pre = readmatrix(name,'FileType','text');
    
    name = sprintf('Preds_post_lstm_%i.out',i);
    lstm_post = readmatrix(name,'FileType','text');

    hz_slowflow_full_lstm_pre=SF2fulltime(t_miss_plot,t_miss_plot,lstm_pre,freqs,N_fmodes);
    hz_slowflow_full_lstm_post=SF2fulltime(t_miss_plot,t_miss_plot,lstm_post,freqs,N_fmodes);
    
    length_miss = length(hz_slowflow_full_lstm_post(1,:));
    for m = 1:length_miss
        hz_slowflow_full_lstm = (1- (m./length_miss))*hz_slowflow_full_lstm_pre + (m/length_miss)*hz_slowflow_full_lstm_post;
    end
    %%%%%%%%%%%%%%%%%%%
    
    %%%%% SINGULAR SPECTRUM ANALYSIS METHOD %%%%%
    name = sprintf('SSA_preds_%i.out',i);
    ssa_full = dlmread(name);
    ssa = ssa_full(t_miss(1)+1:t_miss(end)-1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%% CNN+LSTM %%%%%%%   
    name = sprintf('Preds_pre_cnn+lstm_%i.out',i);
    cnn_lstm_pre = dlmread(name);
    
    name = sprintf('Preds_post_cnn+lstm_%i.out',i);
    cnn_lstm_post = dlmread(name);
    
    length_miss = length(cnn_lstm_post(1,:));
    for m = 1:length_miss
        cnn_lstm = (1- (m/length_miss))*cnn_lstm_pre + (m/length_miss)*cnn_lstm_post;
    end
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
    [cnn_lstm_wave_height ,cnn_lstm_forecast_time,~, ~]=my_wave_height_filter(cnn_lstm, N_pts_per_wave);
    [ssa_wave_height ,ssa_forecast_time,~, ~]=my_wave_height_filter(ssa, N_pts_per_wave);
    %[~,baseline_time,~, ~]=my_wave_height_filter(hz_slowflow_full_baseline, N_pts_per_wave);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%% CALCULATIONS FOR FREQUENCY EXTRACTION %%%%%%%%%
    

    %%%%%%%%% COMPARING OBTAINED PEAKS WITH TRUTH %%%%%%%%
    [h_cmp_fit,t_cmp_fit] = my_wave_height_compare(true_height,true_time,fit_wave_height,fit_time);
    [h_cmp_lstm,t_cmp_lstm] = my_wave_height_compare(true_height,true_time,lstm_wave_height,lstm_forecast_time);
    [h_cmp_cnn_lstm,t_cmp_cnn_lstm] = my_wave_height_compare(true_height,true_time,cnn_lstm_wave_height,cnn_lstm_forecast_time);
    [h_cmp_ssa, t_cmp_ssa] = my_wave_height_compare(true_height,true_time,ssa_wave_height,ssa_forecast_time);
    %[h_cmp_baseline, t_cmp_baseline] = my_wave_height_compare(true_height,true_time,baseline_wave_height,true_time);
    [h_cmp_baseline, t_cmp_baseline]=my_wave_height_compare(true_height,true_time,mean_wave_guess,T_mean.* (0:N_waves_forecast-1));
    [h_cmp_truth, t_cmp_truth] = my_wave_height_compare(true_height,true_time,true_height,true_time);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%% PEAKS TILL 1 MINUTE %%%%%%%%%
    idx_1min=length(t_cmp_truth);
    
    h_cmp_truth_1min = h_cmp_truth;                   % Truth 1 minute peaks
    t_cmp_truth_1min = t_cmp_truth;                   % Truth 1 minute peak times

    h_cmp_baseline_1min = h_cmp_baseline(1:idx_1min); % Baseline 1 minute peaks
    t_cmp_baseline_1min = t_cmp_baseline(1:idx_1min); % Baseline 1 minute peak times

    h_cmp_fit_1min = h_cmp_fit(1:idx_1min);           % Fit 1 minute peaks
    t_cmp_fit_1min = t_cmp_fit(1:idx_1min);           % Fit 1 minute peak times

    h_cmp_lstm_1min = h_cmp_lstm(1:idx_1min);         % LSTM 1 minute peaks
    t_cmp_lstm_1min = t_cmp_lstm(1:idx_1min);         % LSTM 1 minute peak times
    
    h_cmp_cnn_lstm_1min = h_cmp_cnn_lstm(1:idx_1min); % CNN+LSTM 1 minute peaks
    t_cmp_cnn_lstm_1min = t_cmp_cnn_lstm(1:idx_1min); % CNN+LSTM 1 minute peak times
    
    h_cmp_ssa_1min = h_cmp_ssa(1:idx_1min);           % SSA 1 minute peaks
    t_cmp_ssa_1min = t_cmp_ssa(1:idx_1min);           % SSA 1 minute peak times 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%% CUMULATIVE PEAKS FOR ALL THE WINDOWS FOR CORRELATION CALCULATION %%%%%%
    truth_cum = horzcat(truth_cum, h_cmp_truth_1min);
    lstm_cum = horzcat(lstm_cum, h_cmp_lstm_1min);
    cnn_lstm_cum = horzcat(cnn_lstm_cum, h_cmp_cnn_lstm_1min);
    ssa_cum = horzcat(ssa_cum, h_cmp_ssa_1min);
    fit_cum = horzcat(fit_cum, h_cmp_fit_1min);
    baseline_cum = horzcat(baseline_cum, h_cmp_baseline_1min);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%% ERRORS IN MAE %%%%%%%%%%
    err_lstm = [mean(abs(h_cmp_truth_1min - h_cmp_lstm_1min), 'omitnan') mean(abs(t_cmp_truth_1min - t_cmp_lstm_1min), 'omitnan')];
    err_cnn_lstm = [mean(abs(h_cmp_truth_1min - h_cmp_cnn_lstm_1min), 'omitnan') mean(abs(t_cmp_truth_1min - t_cmp_cnn_lstm_1min), 'omitnan')];
    err_ssa = [mean(abs(h_cmp_truth_1min - h_cmp_ssa_1min), 'omitnan') mean(abs(t_cmp_truth_1min - t_cmp_ssa_1min), 'omitnan')];
    err_fit = [mean(abs(h_cmp_truth_1min - h_cmp_fit_1min), 'omitnan') mean(abs(t_cmp_truth_1min - t_cmp_fit_1min), 'omitnan')];
    err_baseline = [mean(abs(h_cmp_truth_1min - h_cmp_baseline_1min), 'omitnan') mean(abs(t_cmp_truth_1min - t_cmp_baseline_1min), 'omitnan')];
    
    LSTM(i,1) = err_lstm(1); 
    CNN_LSTM(i,1) = err_cnn_lstm(1); 
    SSA(i,1) = err_ssa(1); 
    Fit(i,1) = err_fit(1); 
    Baseline(i,1) = err_baseline(1);
    
    t_LSTM(i,1) = err_lstm(2); 
    t_CNN_LSTM(i,1) = err_cnn_lstm(2); 
    t_SSA(i,1) = err_ssa(2); 
    t_Fit(i,1) = err_fit(2); 
    t_Baseline(i,1) = err_baseline(2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%% ERRORS IN MSE %%%%%%%%%%
    err_lstm = [sqrt(mean((h_cmp_truth_1min-h_cmp_lstm_1min).*(h_cmp_truth_1min-h_cmp_lstm_1min), 'omitnan')) immse(t_cmp_truth_1min, t_cmp_lstm_1min)];
    err_cnn_lstm = [sqrt(mean((h_cmp_truth_1min-h_cmp_cnn_lstm_1min).*(h_cmp_truth_1min-h_cmp_cnn_lstm_1min), 'omitnan')) immse(t_cmp_truth_1min, t_cmp_cnn_lstm_1min)];
    err_ssa = [sqrt(mean((h_cmp_truth_1min-h_cmp_ssa_1min).*(h_cmp_truth_1min-h_cmp_ssa_1min), 'omitnan')) immse(t_cmp_truth_1min, t_cmp_ssa_1min)];
    err_fit = [sqrt(mean((h_cmp_truth_1min-h_cmp_fit_1min).*(h_cmp_truth_1min-h_cmp_fit_1min), 'omitnan')) immse(t_cmp_truth_1min, t_cmp_fit_1min)];
    err_baseline = [sqrt(mean((h_cmp_truth_1min-h_cmp_baseline_1min).*(h_cmp_truth_1min-h_cmp_baseline_1min), 'omitnan')) immse(t_cmp_truth_1min, t_cmp_baseline_1min)];
    
    LSTM(i,2) = err_lstm(1); 
    CNN_LSTM(i,2) = err_cnn_lstm(1); 
    SSA(i,2) = err_ssa(1); 
    Fit(i,2) = err_fit(1); 
    Baseline(i,2) = err_baseline(1);
    
    t_LSTM(i,2) = err_lstm(2); 
    t_CNN_LSTM(i,2) = err_cnn_lstm(2); 
    t_SSA(i,2) = err_ssa(2); 
    t_Fit(i,2) = err_fit(2); 
    t_Baseline(i,2) = err_baseline(2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%% ERRORS IN R2 %%%%%%%%%%
    err_lstm = corr(h_cmp_truth_1min', h_cmp_lstm_1min');
    err_cnn_lstm = corr(h_cmp_truth_1min', h_cmp_cnn_lstm_1min');
    err_ssa = corr(h_cmp_truth_1min', h_cmp_ssa_1min');
    err_fit = corr(h_cmp_truth_1min', h_cmp_fit_1min');
    err_baseline = corr(h_cmp_truth_1min', h_cmp_baseline_1min');
    
    LSTM(i,3) = err_lstm; 
    CNN_LSTM(i,3) = err_cnn_lstm; 
    SSA(i,3) = err_ssa; 
    Fit(i,3) = err_fit; 
    Baseline(i,3) = err_baseline;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%% MAXIMUM SURFACE ELEVATIONS %%%%%%
    [max_wave_truth,~]=max(abs(h_miss_plot(1:idx_1min)));
    [max_wave_fit,~]=max(abs(hz_slowflow_full_fit(1:idx_1min)));
    [max_wave_baseline,~]=max(abs(h_cmp_baseline(1:idx_1min)));
    [max_wave_lstm,~]=max(abs(hz_slowflow_full_lstm(1:idx_1min)));
    [max_wave_ssa,~]=max(abs(ssa(1:idx_1min)));
    [max_wave_cnn_lstm,~]=max(abs(cnn_lstm(1:idx_1min)));
    
    Max_LSTM(i) = abs(max_wave_truth - max_wave_lstm);
    Max_CNN_LSTM(i) = abs(max_wave_truth - max_wave_cnn_lstm);
    Max_SSA(i) = abs(max_wave_truth - max_wave_ssa);
    Max_Fit(i) = abs(max_wave_truth - max_wave_fit);
    Max_Baseline(i) = abs(max_wave_truth - max_wave_baseline);
    
    writematrix(LSTM,'LSTM_errors.txt','Delimiter','tab');
    writematrix(CNN_LSTM,'CNN_LSTM_errors.txt','Delimiter','tab');
    writematrix(SSA,'SSA_errors.txt','Delimiter','tab');
    writematrix(Fit,'Fit_errors.txt','Delimiter','tab');
    writematrix(Baseline,'Baseline_errors.txt','Delimiter','tab');
    
    writematrix(t_LSTM,'t_LSTM_errors.txt','Delimiter','tab');
    writematrix(t_CNN_LSTM,'t_CNN_LSTM_errors.txt','Delimiter','tab');
    writematrix(t_SSA,'t_SSA_errors.txt','Delimiter','tab');
    writematrix(t_Fit,'t_Fit_errors.txt','Delimiter','tab');
    writematrix(t_Baseline,'t_Baseline_errors.txt','Delimiter','tab');
    
    writematrix(Max_LSTM,'LSTM_max_errors.txt','Delimiter','tab');
    writematrix(Max_CNN_LSTM,'CNN_LSTM_max_errors.txt','Delimiter','tab');
    writematrix(Max_SSA,'SSA_max_errors.txt','Delimiter','tab');
    writematrix(Max_Fit,'Fit_max_errors.txt','Delimiter','tab');
    writematrix(Max_Baseline,'Baseline_max_errors.txt','Delimiter','tab');
    
    writematrix(lstm_cum,'LSTM_cumulative.txt');
    writematrix(cnn_lstm_cum,'CNN_LSTM_cumulative.txt');
    writematrix(ssa_cum,'SSA_cumulative.txt');
    writematrix(fit_cum,'Fit_cumulative.txt');
    writematrix(baseline_cum,'Baseline_cumulative.txt');
    writematrix(truth_cum,'Truth_cumulative.txt');
    
    disp(i);
end

%lstm = readmatrix('Preds_cnn+lstm_mce.out','FileType','text');
%mse = readmatrix('Preds_cnn+lstm_mse.out','FileType','text');
%newFunc = readmatrix('Preds_cnn+lstm_newFunc.out','FileType','text');

%Freqs = readmatrix('Omegas.csv');

%T_whole = readmatrix('Whole_Time.csv');
%T_pre = readmatrix('Pre_Time.csv');
%T_post = readmatrix('Post_Time.csv');

%fit_whole = readmatrix('Slow_amp_whole.csv');

%fit_pre = readmatrix('Slow_amp_pre.csv');
%fit_post = readmatrix('Slow_amp_post.csv');

%fit_cumulative = horzcat(fit_pre, fit_post);

%t_miss = T_whole(length(T_pre):length(T_pre)+392);
%slow_amp_miss = fit_whole(:,length(fit_pre(1,:)):length(fit_pre(1,:))+392);

%slow_amp_mean = zeros(33, 393);
%for i=1:33
%    slow_amp_mean(i,:) =  repmat(mean(fit_pre(i,:),2),1,393);
%end

%hz_slowflow_full_mce=SF2fulltime(t_miss,t_miss,mce,Freqs,N_fmodes);
%hz_slowflow_full_mse=SF2fulltime(t_miss,t_miss,mse,Freqs,N_fmodes);
%hz_slowflow_full_newFunc=SF2fulltime(t_miss,t_miss,newFunc,Freqs,N_fmodes);

%hz_slowflow_full_fit=SF2fulltime(t_miss,t_miss,slow_amp_miss,Freqs,N_fmodes);
%hz_slowflow_full_baseline=SF2fulltime(t_miss,t_miss,slow_amp_mean,Freqs,N_fmodes);

%Deducting the crest heights and trough depths from the individual
% surface elevation profiles. 
% [fit_wave_height ,fit_time,~, ~]=my_wave_height_filter(hz_slowflow_full_fit, N_pts_per_wave);
% [baseline_wave_height ,baseline_time,~, ~]=my_wave_height_filter(hz_slowflow_full_baseline, N_pts_per_wave);
% 
% [mce_wave_height ,mce_forecast_time,~, ~]=my_wave_height_filter(hz_slowflow_full_mce, N_pts_per_wave);
% [mse_wave_height ,mse_forecast_time,~, ~]=my_wave_height_filter(hz_slowflow_full_mse, N_pts_per_wave);
% [newFunc_wave_height ,newFunc_forecast_time,~, ~]=my_wave_height_filter(hz_slowflow_full_newFunc, N_pts_per_wave);
% 
% [h_cmp_mse,t_cmp_mse] = my_wave_height_compare(fit_wave_height,fit_time,mse_wave_height,mse_forecast_time);
% [h_cmp_mce,t_cmp_mce] = my_wave_height_compare(fit_wave_height,fit_time,mce_wave_height,mce_forecast_time);
% [h_cmp_newFunc,t_cmp_newFunc] = my_wave_height_compare(fit_wave_height,fit_time,newFunc_wave_height,newFunc_forecast_time);
% 
% [h_cmp_baseline, t_cmp_baseline] = my_wave_height_compare(fit_wave_height,fit_time,baseline_wave_height,baseline_time);
% [h_cmp_fit, t_cmp_fit] = my_wave_height_compare(fit_wave_height,fit_time,fit_wave_height,fit_time);


%%%%%% 1 min errors and plots %%%%%%
% idx_1min=find(fit_time > 60,1)-1;
% h_cmp_baseline_1min = [h_cmp_baseline(1:idx_1min)];
% t_cmp_baseline_1min = [t_cmp_baseline(1:idx_1min)];
% 
% h_cmp_fit_1min = [h_cmp_fit(1:idx_1min)];
% t_cmp_fit_1min = [t_cmp_fit(1:idx_1min)];
% 
% h_cmp_mse_1min = [h_cmp_mse(1:idx_1min)];
% h_cmp_mce_1min = [h_cmp_mce(1:idx_1min)];
% h_cmp_newFunc_1min = [h_cmp_newFunc(1:idx_1min)];
% 
% t_cmp_mse_1min = [t_cmp_mse(1:idx_1min)];
% t_cmp_mce_1min = [t_cmp_mce(1:idx_1min)];
% t_cmp_newFunc_1min = [t_cmp_newFunc(1:idx_1min)];

% err_pred_mse = [mean(abs(h_cmp_fit_1min - h_cmp_mse_1min)) mean(abs(t_cmp_fit_1min - t_cmp_mse_1min))];
% err_pred_mce = [mean(abs(h_cmp_fit_1min - h_cmp_mce_1min)) mean(abs(t_cmp_fit_1min - t_cmp_mce_1min))];
% err_pred_newFunc = [mean(abs(h_cmp_fit_1min - h_cmp_newFunc_1min)) mean(abs(t_cmp_fit_1min - t_cmp_newFunc_1min))];
% err_baseline = [mean(abs(h_cmp_fit_1min - h_cmp_baseline_1min)) mean(abs(t_cmp_fit_1min - t_cmp_baseline_1min))];

% std_pred_mse = [std(abs(h_cmp_fit_1min - h_cmp_mse_1min)) mean(abs(t_cmp_fit_1min - t_cmp_mse_1min))];
% std_pred_mce = [std(abs(h_cmp_fit_1min - h_cmp_mce_1min)) mean(abs(t_cmp_fit_1min - t_cmp_mce_1min))];
% std_pred_newFunc = [std(abs(h_cmp_fit_1min - h_cmp_newFunc_1min)) mean(abs(t_cmp_fit_1min - t_cmp_newFunc_1min))];
% std_baseline = [std(abs(h_cmp_fit_1min - h_cmp_baseline_1min)) mean(abs(t_cmp_fit_1min - t_cmp_baseline_1min))];

% [max_wave_fit,max_wave_fit_idx]=max(abs(hz_slowflow_full_fit(1:idx_1min)));
% [max_wave_baseline,max_wave_baseline_idx]=max(abs(hz_slowflow_full_baseline(1:idx_1min)));
% [max_wave_mse,max_wave_mse_idx]=max(abs(hz_slowflow_full_mce(1:idx_1min)));
% [max_wave_mce,max_wave_mce_idx]=max(abs(hz_slowflow_full_mse(1:idx_1min)));
% [max_wave_newFunc,max_wave_newFunc_idx]=max(abs(hz_slowflow_full_newFunc(1:idx_1min)));

% err_highest_mse = abs(max_wave_fit - max_wave_mse);
% err_highest_mce = abs(max_wave_fit - max_wave_mce);
% err_highest_newFunc = abs(max_wave_fit - max_wave_newFunc);
% err_highest_baseline = abs(max_wave_fit - max_wave_baseline);

% arrs = [LSTM(:,1) CNN_LSTM(:,1)  SSA(:,1)  Fit(:,1)  Baseline(:,1)];
% %arrs = [LSTM(:,1) SSA(:,1) Fit(:,1) Baseline(:,1)];
% 
% arrs_time = [t_LSTM(:,1)  t_CNN_LSTM(:,1)  t_SSA(:,1)  t_Fit(:,1)  t_Baseline(:,1)];
% %arrs_time = [t_LSTM(:,1) t_SSA(:,1) t_Fit(:,1) t_Baseline(:,1)];
% num_arrs = length(arrs(1,:));
% plotBarCats(arrs, arrs_time, num_arrs, mins)

% mean_error_height = [err_pred_mse(1), err_pred_mce(1), err_pred_newFunc(1), err_baseline(1)];
% std_error_height = [std_pred_mse(1), std_pred_mce(1), std_pred_newFunc(1), std_baseline(1)];
% mean_error_time = [err_pred_mse(2), err_pred_mce(2), err_pred_newFunc(2), err_baseline(2)];
% std_error_time = [std_pred_mse(2), std_pred_mce(2), std_pred_newFunc(2), std_baseline(2)];

% X = categorical({'MSE','MCE','New function','Benchmark'});
% figure
% g = gcf;
% subplot(1,2,1)
% h = bar(X,diag(mean_error_height),'stacked');
% ylabel('average error in m','interpreter','latex')
% subplot(1,2,2)
% f = bar(X,diag(mean_error_time),'stacked');
% ylabel('average error in sec','interpreter','latex')
% sgtitle('Errors for 1 minute for cnn+lstm for different loss functions','interpreter','latex','fontweight','normal','fontsize',12);
% 
% print(g,'Errors for 1min for cnn+lstm.tiff','-dtiff','-r300'); 

% highest_abs_error = [err_highest_mse err_highest_mce err_highest_newFunc err_highest_baseline];
% figure;
% g = gcf;
% bar(X,diag(highest_abs_error),'stacked')
% ylabel('average maximal height/depths error (m)','interpreter','latex')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 8 8];
% title('Highest wave errors for cnn+lstm for different loss functions','interpreter','latex','fontweight','normal','fontsize',12);
% print(g,'Errors for highest wave in 1min for cnn+lstm.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_mse_1min.','MSE: CNN+LSTM')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 1min for cnn+lstm_mse.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_mce_1min.','MCE: CNN+LSTM')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 1min for cnn+lstm_mce.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_newFunc_1min.','New Function: CNN+LSTM')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 1min for cnn+lstm_newFunction.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_baseline_1min.','Benchmark')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 1min for cnn+lstm_benchmark.tiff','-dtiff','-r300');

% figure;
% plot(hz_slowflow_full_fit(1:idx_1min),'k-.','marker','s','markerSize',6,'Linewidth',1.0)
% hold on;
% plot(hz_slowflow_full_baseline(1:idx_1min),'b-.','marker','s','markerSize',6,'Linewidth',1.0)
% plot(hz_slowflow_full_mse(1:idx_1min),'r-.','marker','s','markerSize',6,'Linewidth',1.0)
% plot(hz_slowflow_full_mce(1:idx_1min),'g-.','marker','s','markerSize',6,'Linewidth',1.0)
% plot(hz_slowflow_full_newFunc(1:idx_1min),'m-.','marker','s','markerSize',6,'Linewidth',1.0)
% legend('Fit','Baseline','MSE','MCE','New Function','FontName','Times','Location','best','Orientation','vertical') 
% xlabel('Duration of missing data - number of peaks','FontName','Times','FontSize',11)
% ylabel('\eta','FontName','Times','FontSize',11)
% title("Eta for 1 min",'FontName','Times','FontSize',12, 'Fontweight','normal');
% ax = gca;
% ax.XTickMode = 'manual';
% ax.YTickMode = 'manual';
% ax.ZTickMode = 'manual';
% ax.XLimMode = 'manual';
% ax.YLimMode = 'manual';
% ax.ZLimMode = 'manual';
% g = gcf;
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 12 5];
% name = sprintf('CNN_LSTM_1min.png');
%saveas(g,name)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% 2 min errors and plots %%%%%%
% idx_1min=find(fit_time > 2*60,1)-1;
% h_cmp_baseline_1min = [h_cmp_baseline(1:idx_1min)];
% t_cmp_baseline_1min = [t_cmp_baseline(1:idx_1min)];
% 
% h_cmp_fit_1min = [h_cmp_fit(1:idx_1min)];
% t_cmp_fit_1min = [t_cmp_fit(1:idx_1min)];
% 
% h_cmp_mse_1min = [h_cmp_mse(1:idx_1min)];
% h_cmp_mce_1min = [h_cmp_mce(1:idx_1min)];
% h_cmp_newFunc_1min = [h_cmp_newFunc(1:idx_1min)];
% 
% t_cmp_mse_1min = [t_cmp_mse(1:idx_1min)];
% t_cmp_mce_1min = [t_cmp_mce(1:idx_1min)];
% t_cmp_newFunc_1min = [t_cmp_newFunc(1:idx_1min)];
% 
% err_pred_mse = [mean(abs(h_cmp_fit_1min - h_cmp_mse_1min)) mean(abs(t_cmp_fit_1min - t_cmp_mse_1min))];
% err_pred_mce = [mean(abs(h_cmp_fit_1min - h_cmp_mce_1min)) mean(abs(t_cmp_fit_1min - t_cmp_mce_1min))];
% err_pred_newFunc = [mean(abs(h_cmp_fit_1min - h_cmp_newFunc_1min)) mean(abs(t_cmp_fit_1min - t_cmp_newFunc_1min))];
% err_baseline = [mean(abs(h_cmp_fit_1min - h_cmp_baseline_1min)) mean(abs(t_cmp_fit_1min - t_cmp_baseline_1min))];
% 
% std_pred_mse = [std(abs(h_cmp_fit_1min - h_cmp_mse_1min)) mean(abs(t_cmp_fit_1min - t_cmp_mse_1min))];
% std_pred_mce = [std(abs(h_cmp_fit_1min - h_cmp_mce_1min)) mean(abs(t_cmp_fit_1min - t_cmp_mce_1min))];
% std_pred_newFunc = [std(abs(h_cmp_fit_1min - h_cmp_newFunc_1min)) mean(abs(t_cmp_fit_1min - t_cmp_newFunc_1min))];
% std_baseline = [std(abs(h_cmp_fit_1min - h_cmp_baseline_1min)) mean(abs(t_cmp_fit_1min - t_cmp_baseline_1min))];
% 
% [max_wave_fit,max_wave_fit_idx]=max(abs(hz_slowflow_full_fit(1:idx_1min)));
% [max_wave_baseline,max_wave_baseline_idx]=max(abs(hz_slowflow_full_baseline(1:idx_1min)));
% [max_wave_mse,max_wave_mse_idx]=max(abs(hz_slowflow_full_mce(1:idx_1min)));
% [max_wave_mce,max_wave_mce_idx]=max(abs(hz_slowflow_full_mse(1:idx_1min)));
% [max_wave_newFunc,max_wave_newFunc_idx]=max(abs(hz_slowflow_full_newFunc(1:idx_1min)));
% 
% err_highest_mse = abs(max_wave_fit - max_wave_mse);
% err_highest_mce = abs(max_wave_fit - max_wave_mce);
% err_highest_newFunc = abs(max_wave_fit - max_wave_newFunc);
% err_highest_baseline = abs(max_wave_fit - max_wave_baseline);
% 
% mean_error_height = [err_pred_mse(1), err_pred_mce(1), err_pred_newFunc(1), err_baseline(1)];
% std_error_height = [std_pred_mse(1), std_pred_mce(1), std_pred_newFunc(1), std_baseline(1)];
% mean_error_time = [err_pred_mse(2), err_pred_mce(2), err_pred_newFunc(2), err_baseline(2)];
% std_error_time = [std_pred_mse(2), std_pred_mce(2), std_pred_newFunc(2), std_baseline(2)];
% 
% X = categorical({'MSE','MCE','New function','Benchmark'});
% figure
% g = gcf;
% subplot(1,2,1)
% h = bar(X,diag(mean_error_height),'stacked');
% ylabel('average error in m','interpreter','latex')
% subplot(1,2,2)
% f = bar(X,diag(mean_error_time),'stacked');
% ylabel('average error in sec','interpreter','latex')
% sgtitle('Errors for 2 minute for cnn+lstm for different loss functions','interpreter','latex','fontweight','normal','fontsize',12);
% 
% print(g,'Errors for 2min for cnn+lstm.tiff','-dtiff','-r300'); 
% 
% highest_abs_error = [err_highest_mse err_highest_mce err_highest_newFunc err_highest_baseline];
% figure;
% g = gcf;
% bar(X,diag(highest_abs_error),'stacked')
% ylabel('average maximal height/depths error (m)','interpreter','latex')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 8 8];
% title('Highest wave errors for cnn+lstm for different loss functions','interpreter','latex','fontweight','normal','fontsize',12);
% print(g,'Errors for highest wave in 2min for cnn+lstm.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_mse_1min.','MSE: CNN+LSTM - 2minutes')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 2min for cnn+lstm_mse.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_mce_1min.','MCE: CNN+LSTM - 2minutes')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 2min for cnn+lstm_mce.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_newFunc_1min.','New Function: CNN+LSTM - 2minutes')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 2min for cnn+lstm_newFunction.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_baseline_1min.','Benchmark - 2minutes')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 2min for cnn+lstm_benchmark.tiff','-dtiff','-r300');
% 
% % figure;
% % plot(hz_slowflow_full_fit(1:idx_1min),'k-.','marker','s','markerSize',6,'Linewidth',1.0)
% % hold on;
% % plot(hz_slowflow_full_baseline(1:idx_1min),'b-.','marker','s','markerSize',6,'Linewidth',1.0)
% % plot(hz_slowflow_full_mse(1:idx_1min),'r-.','marker','s','markerSize',6,'Linewidth',1.0)
% % plot(hz_slowflow_full_mce(1:idx_1min),'g-.','marker','s','markerSize',6,'Linewidth',1.0)
% % plot(hz_slowflow_full_newFunc(1:idx_1min),'m-.','marker','s','markerSize',6,'Linewidth',1.0)
% % legend('Fit','Baseline','MSE','MCE','New Function','FontName','Times','Location','best','Orientation','vertical') 
% % xlabel('Duration of missing data - number of peaks','FontName','Times','FontSize',11)
% % ylabel('\eta','FontName','Times','FontSize',11)
% % title("Eta for 2 min",'FontName','Times','FontSize',12, 'Fontweight','normal');
% % ax = gca;
% % ax.XTickMode = 'manual';
% % ax.YTickMode = 'manual';
% % ax.ZTickMode = 'manual';
% % ax.XLimMode = 'manual';
% % ax.YLimMode = 'manual';
% % ax.ZLimMode = 'manual';
% % g = gcf;
% % g.PaperUnits = 'inches';
% % g.PaperPosition = [0 0 12 5];
% % name = sprintf('CNN_LSTM_2min.png');
% % saveas(g,name)
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %%%%%% 5 min errors and plots %%%%%%
% idx_1min=find(fit_time > 5*60,1)-1;
% h_cmp_baseline_1min = [h_cmp_baseline(1:idx_1min)];
% t_cmp_baseline_1min = [t_cmp_baseline(1:idx_1min)];
% 
% h_cmp_fit_1min = [h_cmp_fit(1:idx_1min)];
% t_cmp_fit_1min = [t_cmp_fit(1:idx_1min)];
% 
% h_cmp_mse_1min = [h_cmp_mse(1:idx_1min)];
% h_cmp_mce_1min = [h_cmp_mce(1:idx_1min)];
% h_cmp_newFunc_1min = [h_cmp_newFunc(1:idx_1min)];
% 
% t_cmp_mse_1min = [t_cmp_mse(1:idx_1min)];
% t_cmp_mce_1min = [t_cmp_mce(1:idx_1min)];
% t_cmp_newFunc_1min = [t_cmp_newFunc(1:idx_1min)];
% 
% err_pred_mse = [mean(abs(h_cmp_fit_1min - h_cmp_mse_1min)) mean(abs(t_cmp_fit_1min - t_cmp_mse_1min))];
% err_pred_mce = [mean(abs(h_cmp_fit_1min - h_cmp_mce_1min)) mean(abs(t_cmp_fit_1min - t_cmp_mce_1min))];
% err_pred_newFunc = [mean(abs(h_cmp_fit_1min - h_cmp_newFunc_1min)) mean(abs(t_cmp_fit_1min - t_cmp_newFunc_1min))];
% err_baseline = [mean(abs(h_cmp_fit_1min - h_cmp_baseline_1min)) mean(abs(t_cmp_fit_1min - t_cmp_baseline_1min))];
% 
% std_pred_mse = [std(abs(h_cmp_fit_1min - h_cmp_mse_1min)) mean(abs(t_cmp_fit_1min - t_cmp_mse_1min))];
% std_pred_mce = [std(abs(h_cmp_fit_1min - h_cmp_mce_1min)) mean(abs(t_cmp_fit_1min - t_cmp_mce_1min))];
% std_pred_newFunc = [std(abs(h_cmp_fit_1min - h_cmp_newFunc_1min)) mean(abs(t_cmp_fit_1min - t_cmp_newFunc_1min))];
% std_baseline = [std(abs(h_cmp_fit_1min - h_cmp_baseline_1min)) mean(abs(t_cmp_fit_1min - t_cmp_baseline_1min))];
% 
% [max_wave_fit,max_wave_fit_idx]=max(abs(hz_slowflow_full_fit(1:idx_1min)));
% [max_wave_baseline,max_wave_baseline_idx]=max(abs(hz_slowflow_full_baseline(1:idx_1min)));
% [max_wave_mse,max_wave_mse_idx]=max(abs(hz_slowflow_full_mce(1:idx_1min)));
% [max_wave_mce,max_wave_mce_idx]=max(abs(hz_slowflow_full_mse(1:idx_1min)));
% [max_wave_newFunc,max_wave_newFunc_idx]=max(abs(hz_slowflow_full_newFunc(1:idx_1min)));
% 
% err_highest_mse = abs(max_wave_fit - max_wave_mse);
% err_highest_mce = abs(max_wave_fit - max_wave_mce);
% err_highest_newFunc = abs(max_wave_fit - max_wave_newFunc);
% err_highest_baseline = abs(max_wave_fit - max_wave_baseline);
% 
% mean_error_height = [err_pred_mse(1), err_pred_mce(1), err_pred_newFunc(1), err_baseline(1)];
% std_error_height = [std_pred_mse(1), std_pred_mce(1), std_pred_newFunc(1), std_baseline(1)];
% mean_error_time = [err_pred_mse(2), err_pred_mce(2), err_pred_newFunc(2), err_baseline(2)];
% std_error_time = [std_pred_mse(2), std_pred_mce(2), std_pred_newFunc(2), std_baseline(2)];
% 
% X = categorical({'MSE','MCE','New function','Benchmark'});
% figure
% g = gcf;
% subplot(1,2,1)
% h = bar(X,diag(mean_error_height),'stacked');
% ylabel('average error in m','interpreter','latex')
% subplot(1,2,2)
% f = bar(X,diag(mean_error_time),'stacked');
% ylabel('average error in sec','interpreter','latex')
% sgtitle('Errors for 5 minutes for cnn+lstm for different loss functions','interpreter','latex','fontweight','normal','fontsize',12);
% 
% print(g,'Errors for 5min for cnn+lstm.tiff','-dtiff','-r300'); 
% 
% highest_abs_error = [err_highest_mse err_highest_mce err_highest_newFunc err_highest_baseline];
% figure;
% g = gcf;
% bar(X,diag(highest_abs_error),'stacked')
% ylabel('average maximal height/depths error (m)','interpreter','latex')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 8 8];
% title('Highest wave errors for cnn+lstm for different loss functions','interpreter','latex','fontweight','normal','fontsize',12);
% print(g,'Errors for highest wave in 5min for cnn+lstm.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_mse_1min.','MSE: CNN+LSTM - 5 minutes')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 5min for cnn+lstm_mse.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_mce_1min.','MCE: CNN+LSTM - 5 minutes')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 5min for cnn+lstm_mce.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_newFunc_1min.','New Function: CNN+LSTM - 5 minutes')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 5min for cnn+lstm_newFunction.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_baseline_1min.','Benchmark - 5 minutes')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 5min for cnn+lstm_benchmark.tiff','-dtiff','-r300');
% 
% 
% figure;
% plot(hz_slowflow_full_fit,'k-','Linewidth',0.5)
% hold on;
% plot(hz_slowflow_full_baseline,'b-','Linewidth',0.5)
% plot(hz_slowflow_full_mse,'r-','Linewidth',0.5)
% plot(hz_slowflow_full_mce,'g-','Linewidth',0.5)
% plot(hz_slowflow_full_newFunc,'m-','Linewidth',0.5)
% legend('Fit','Baseline','MSE','MCE','New Function','FontName','Times','Location','best','Orientation','vertical') 
% xlabel('Duration of missing data','FontName','Times','FontSize',11)
% ylabel('\eta','FontName','Times','FontSize',11)
% ax = gca;
% ax.XTickMode = 'manual';
% ax.YTickMode = 'manual';
% ax.ZTickMode = 'manual';
% ax.XLimMode = 'manual';
% ax.YLimMode = 'manual';
% ax.ZLimMode = 'manual';
% g = gcf;
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 12 4];
% name = sprintf('CNN_LSTM_etas.png');
% saveas(g,name)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all