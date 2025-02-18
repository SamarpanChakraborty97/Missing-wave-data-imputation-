% clear all
% close all
% clc
% 
% samplerate = 1.2800;
% 
% % src='067p1_d26.nc';
% % 
% % ncdisp(src)
% % 
% % info=ncinfo(src,'xyzZDisplacement');
% % Starttime=ncread(src,'xyzStartTime');
% % samplerate=ncread(src,'xyzSampleRate');
% % filt_delay=ncread(src,'xyzFilterDelay');
% % BeginDate=datetime(1970,1,1)+seconds(Starttime-filt_delay);
% % EndDate=datetime(1970,1,1)+seconds(Starttime-filt_delay)+seconds(info.Size/samplerate);
% % 
% eps=1*10^-3; % Set the parameter epsilon (cf. equation (1) and (2))
% alpha = 0.5;
% N_pts_per_wave=1;
% N_windows=100; % Number of windows for comparison
% N_MC=1000; % Number of Monte Carlo samples while solving the nonlinear function optimization for frequencies (4)
% N_fmodes=16; % Number of elementary waves/ Fourier modes, i.e. parameter Nf in equation % (1)
% % 
% % % Number of waves in the inital fitting interval. For the given
% % % observations 55 wave corresponds to approximately 15 min of observations.
% % 
% % num_minutes = 16;
% % num_miss_minutes = 7;
% % num_total_minutes = (num_minutes)*2 + num_miss_minutes;
% % 
% % window_size_pre = ceil(num_minutes * 60 * samplerate);
% % window_size_post = ceil(num_minutes * 60 * samplerate);
% % window_size_miss = ceil(num_miss_minutes * 60 * samplerate);
% % window_size_tot = window_size_pre + window_size_post + window_size_miss;
% % 
% % h_tmp=cell(3,1);
% % t_tmp=cell(3,1);
% % hz_slowflow=cell(3,1);
% % slow_vars=cell(3,1);
% % 
% % % select a random start time
% % w_start=randi(info.Size-window_size_tot);
% % 
% % hz=ncread(src,'xyzZDisplacement',w_start,window_size_tot).';
% % 
% % % extract extreme values, i.e. trough depths and crests heights
% % [tmp_wave_height ,tmp_wave_idx,zero_idx, ~]=my_wave_height_filter(hz ,N_pts_per_wave);
% % 
% % ts=[tmp_wave_idx];%  tmp_t0];%[1:window_size];%
% % hs=[tmp_wave_height];% zeros(1,length(zero_idx))];%hz(jj,:);%
% % 
% % [ts,id]=sort(ts);
% % hs=hs(id);
% % %missing_len = ceil(samplerate * 60); %1 minute of missing data
% % 
% % N_waves_tot = length(hs);
% % N_waves_miss = 10 * num_miss_minutes;
% % N_waves_pre = ceil((length(hs)-N_waves_miss)/2);
% % N_waves_post = N_waves_tot - N_waves_miss - N_waves_pre;
% % 
% % t_full_peaks=ts(1:N_waves_tot);
% % h_full_peaks=hs(1:N_waves_tot);
% % 
% % t_pre_peaks=ts(1:N_waves_pre);
% % h_pre_peaks=hs(1:N_waves_pre);
% % 
% % t_miss_peaks = ts(N_waves_pre+1:N_waves_pre+N_waves_miss)
% % h_miss_peaks = hs(N_waves_pre+1:N_waves_pre+N_waves_miss);
% % 
% % t_post_peaks=ts(N_waves_pre+N_waves_miss+1:N_waves_tot);
% % h_post_peaks=hs(N_waves_pre+N_waves_miss+1:N_waves_tot);
% 
% % t_full_plot = (1:t_full(end));
% % h_full_plot = hz(1:t_full(end));
% % 
% % t_pre_plot = (t_pre(1):t_pre(end));
% % h_pre_plot = hz(t_pre(1):t_pre(end));
% % 
% % t_post_plot = (t_post(1):t_post(end));
% % h_post_plot = hz(t_post(1):t_post(end));
% % 
% % t_miss_plot = (t_miss(1):t_miss(end));
% % h_miss_plot = hz(t_miss(1):t_miss(end));
% 
% mce = readmatrix('Preds_LSTM_MCE.out','FileType','text');
% mse = readmatrix('Preds_LSTM_MSE.out','FileType','text');
% newFunc = readmatrix('Preds_LSTM_newFunc.out','FileType','text');
% 
% Freqs = readmatrix('Omegas.csv');
% 
% T_whole = readmatrix('Whole_Time.csv');
% T_pre = readmatrix('Pre_Time.csv');
% T_post = readmatrix('Post_Time.csv');
% 
% fit_whole = readmatrix('Slow_amp_whole.csv');
% 
% fit_pre = readmatrix('Slow_amp_pre.csv');
% fit_post = readmatrix('Slow_amp_post.csv');
% 
% fit_cumulative = horzcat(fit_pre, fit_post);
% 
% t_miss = T_whole(length(T_pre):length(T_pre)+392);
% slow_amp_miss = fit_whole(:,length(fit_pre(1,:)):length(fit_pre(1,:))+392);
% 
% slow_amp_mean = zeros(33, 393);
% for i=1:33
%     slow_amp_mean(i,:) =  repmat(mean(fit_pre(i,:),2),1,393);
% end
% 
% hz_slowflow_full_mce=SF2fulltime(t_miss,t_miss,mce,Freqs,N_fmodes);
% hz_slowflow_full_mse=SF2fulltime(t_miss,t_miss,mse,Freqs,N_fmodes);
% hz_slowflow_full_newFunc=SF2fulltime(t_miss,t_miss,newFunc,Freqs,N_fmodes);
% 
% hz_slowflow_full_fit=SF2fulltime(t_miss,t_miss,slow_amp_miss,Freqs,N_fmodes);
% hz_slowflow_full_baseline=SF2fulltime(t_miss,t_miss,slow_amp_mean,Freqs,N_fmodes);
% 
% %Deducting the crest heights and trough depths from the individual
% % surface elevation profiles. 
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
% 
% 
% %%%%%% 1 min errors and plots %%%%%%
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
% sgtitle('Errors for 1 minute for lstm for different loss functions','interpreter','latex','fontweight','normal','fontsize',12);
% 
% print(g,'Errors for 1min for lstm.tiff','-dtiff','-r300'); 
% 
% highest_abs_error = [err_highest_mse err_highest_mce err_highest_newFunc err_highest_baseline];
% figure;
% g = gcf;
% bar(X,diag(highest_abs_error),'stacked')
% ylabel('average maximal height/depths error (m)','interpreter','latex')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 8 8];
% title('Highest wave errors for lstm for different loss functions','interpreter','latex','fontweight','normal','fontsize',12);
% print(g,'Errors for highest wave in 1min for lstm.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_mse_1min.','MSE: LSTM')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 1min for lstm_mse.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_mce_1min.','MCE: LSTM')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 1min for lstm_mce.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_newFunc_1min.','New Function: LSTM')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 1min for lstm_newFunction.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_baseline_1min.','Benchmark')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 1min for lstm_benchmark.tiff','-dtiff','-r300');
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %%%%%% 2 min errors and plots %%%%%%
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
% sgtitle('Errors for 2 minute for lstm for different loss functions','interpreter','latex','fontweight','normal','fontsize',12);
% 
% print(g,'Errors for 2min for lstm.tiff','-dtiff','-r300'); 
% 
% highest_abs_error = [err_highest_mse err_highest_mce err_highest_newFunc err_highest_baseline];
% figure;
% g = gcf;
% bar(X,diag(highest_abs_error),'stacked')
% ylabel('average maximal height/depths error (m)','interpreter','latex')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 8 8];
% title('Highest wave errors for cnn+lstm for different loss functions','interpreter','latex','fontweight','normal','fontsize',12);
% print(g,'Errors for highest wave in 2min for lstm.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_mse_1min.','MSE: LSTM - 2minutes')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 2min for lstm_mse.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_mce_1min.','MCE: LSTM - 2minutes')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 2min for lstm_mce.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_newFunc_1min.','New Function: LSTM - 2minutes')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 2min for lstm_newFunction.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_baseline_1min.','Benchmark - 2minutes')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 2min for lstm_benchmark.tiff','-dtiff','-r300');
% 
% 
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
% sgtitle('Errors for 5 minutes for lstm for different loss functions','interpreter','latex','fontweight','normal','fontsize',12);
% 
% print(g,'Errors for 5min for lstm.tiff','-dtiff','-r300'); 
% 
% highest_abs_error = [err_highest_mse err_highest_mce err_highest_newFunc err_highest_baseline];
% figure;
% g = gcf;
% bar(X,diag(highest_abs_error),'stacked')
% ylabel('average maximal height/depths error (m)','interpreter','latex')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 8 8];
% title('Highest wave errors for lstm for different loss functions','interpreter','latex','fontweight','normal','fontsize',12);
% print(g,'Errors for highest wave in 5min for lstm.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_mse_1min.','MSE: LSTM - 5 minutes')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 5min for lstm_mse.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_mce_1min.','MCE: LSTM - 5 minutes')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 5min for lstm_mce.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_newFunc_1min.','New Function: LSTM - 5 minutes')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 5min for lstm_newFunction.tiff','-dtiff','-r300');
% 
% figure
% g = gcf;
% my_scatter2(h_cmp_fit_1min.',h_cmp_baseline_1min.','Benchmark - 5 minutes')
% g.PaperUnits = 'inches';
% g.PaperPosition = [0 0 6 6];
% print(g,'Correlation plots for 5min for lstm_benchmark.tiff','-dtiff','-r300');


figure;
plot(hz_slowflow_full_fit,'k-','Linewidth',0.5)
hold on;
plot(hz_slowflow_full_baseline,'b-','Linewidth',0.5)
plot(hz_slowflow_full_mse,'r-','Linewidth',0.5)
plot(hz_slowflow_full_mce,'g-','Linewidth',0.5)
plot(hz_slowflow_full_newFunc,'m-','Linewidth',0.5)
legend('Fit','Baseline','MSE','MCE','New Function','FontName','Times','Location','best','Orientation','vertical') 
xlabel('Duration of missing data','FontName','Times','FontSize',11)
ylabel('\eta','FontName','Times','FontSize',11)
ax = gca;
ax.XTickMode = 'manual';
ax.YTickMode = 'manual';
ax.ZTickMode = 'manual';
ax.XLimMode = 'manual';
ax.YLimMode = 'manual';
ax.ZLimMode = 'manual';
g = gcf;
g.PaperUnits = 'inches';
g.PaperPosition = [0 0 12 4];
name = sprintf('LSTM_etas.png');
saveas(g,name)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all