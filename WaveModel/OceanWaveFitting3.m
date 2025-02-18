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

num_minutes = 1;
num_miss_minutes = 0.5;
num_total_minutes = (num_minutes)*2 + num_miss_minutes;

window_size_pre = ceil(num_minutes * 60 * samplerate);
window_size_post = ceil(num_minutes * 60 * samplerate);
window_size_miss = ceil(num_miss_minutes * 60 * samplerate);
window_size_tot = window_size_pre + window_size_post + window_size_miss;

Num_Windows = 100;
initial_window_start = 5;



% select a random start time
w_start = initial_window_start : 500 : (500*Num_Windows)+initial_window_start;
for i=1:length(w_start)
    
    h_tmp=cell(3, 1);
    t_tmp=cell(3, 1);
    hz_slowflow=cell(3, 1);
    slow_vars=cell(3, 1);

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
    
    t_miss=ts(N_waves_pre+1:N_waves_pre+N_waves_miss);
    
    t_post=ts(N_waves_pre+N_waves_miss+1:N_waves_tot);
    h_post=hs(N_waves_pre+N_waves_miss+1:N_waves_tot);
    
    t_full_plot = (1:t_full(end));
    h_full_plot = hz(1:t_full(end));
    
    t_pre_plot = (t_pre(1):t_pre(end));
    h_pre_plot = hz(t_pre(1):t_pre(end));
    
    t_post_plot = (t_post(1):t_post(end));
    h_post_plot = hz(t_post(1):t_post(end));
    
    t_miss_plot = (t_miss(1):t_miss(end));
    h_miss_plot = hz(t_miss(1):t_miss(end));
    
    if N_fmodes<9
        % If less than nine Fourier modes are selected a MATLAB routine can be
        % used for fitting.
        str_fmode=append('sin',num2str(N_fmodes));
        my_fit=fit(t_pre.',(h_pre-mean(h_pre)).',str_fmode);
        err_const=sum((h_pre-mean(h_pre)-my_fit(t_pre).').^2);
        coeff2=coeffvalues(my_fit);
        ws=sort(coeff2(2:3:end));
        tic
        % Monte Carlo sampling for optimization (4)
        [ws_MC, ~]= my_Freq_MC_fit(t_pre,h_pre,err_const,N_fmodes,ws,N_MC,samplerate);
        toc
    else
        ws=zeros(1,N_fmodes);
        err_const=sum((h_pre-mean(h_pre)).^2);
        tic
        % Monte Carlo sampling for optimization (4)
        [ws_MC, ~]= my_Freq_MC_fit2(t_pre,h_pre,err_const,N_fmodes,ws,N_MC,samplerate);
        toc
    end
    
    % Fitting the slowly varying amplitudes, i.e. solving the linear
    % optimization (5), for the whole duration (cf. Fig. 3).
    tic
    h_tmp{1}= h_full;
    t_tmp{1}= t_full;
    [hz_slowflow{1} ,slow_vars{1}, ~]= my_SF_fit(t_full, h_full,ws_MC,eps,alpha);
    toc
    
    % Utilize the slowly varying amplitudes from the whole time interval to
    % fit the dynamical systems to be used for comparisons (cf. Fig. 3).
    slow_vars_fit=slow_vars{1};
    tt=t_tmp{1}(1):t_tmp{1}(end);
    
    % Interpolation yielding uniform time series for the slowly varying amplitudes.
    slow_vars_interp_full =interp1(t_tmp{1},slow_vars_fit.',tt,'spline','extrap').';
    
    % Fitting the slowly varying amplitudes, i.e. solving the linear
    % optimization (5) for 1.5 minutes before the mssing data(cf. Fig. 3).
    tic
    t_tmp{2}= t_pre;
    h_tmp{2}= h_pre;
    [hz_slowflow{2} ,slow_vars{2},~]= my_SF_fit(t_tmp{2}, h_tmp{2},ws_MC,eps,alpha);
    toc
    
    % Utilize the slowly varying amplitudes from the 1.5 minutes before the missing data to
    % fit the dynamical systems to be used for comparisons (cf. Fig. 3).
    slow_vars_fit=slow_vars{2};
    tt=t_tmp{2}(1):t_tmp{2}(end);
    
    % Interpolation yielding uniform time series for the slowly varying amplitudes.
    slow_vars_interp_pre =interp1(t_tmp{2},slow_vars_fit.',tt,'spline','extrap').';
    
    % Fitting the slowly varying amplitudes, i.e. solving the linear
    % optimization (5) for the 1.5 minutes after the missinng data (cf. Fig. 3).
    tic
    h_tmp{3}= h_post;
    t_tmp{3}= t_post;
    [hz_slowflow{3} ,slow_vars{3},~]= my_SF_fit(t_tmp{3}, h_tmp{3},ws_MC,eps, alpha);
    toc
    
    % Utilize the slowly varying amplitudes from the 1.5 minutes after the missing data to
    % fit the dynamical systems to be used for comparisons (cf. Fig. 3).
    slow_vars_fit=slow_vars{3};
    tt=t_tmp{3}(1):t_tmp{3}(end);
    
    % Interpolation yielding uniform time series for the slowly varying amplitudes.
    slow_vars_interp_post =interp1(t_tmp{3},slow_vars_fit.',tt,'spline','extrap').';
    
    %t_tmp{1}(1):t_tmp{1}(end)
    %t_tmp{2}(1):t_tmp{2}(end)
    %t_tmp{3}(1):t_tmp{3}(end)
    
    %ws_MC
    
    %m0= mean(slow_vars_interp,2);
    %m1=std(slow_vars_interp,0,2);
    %slow_vars_nomean=(slow_vars_interp-m0)./m1;
    
    name_amp = sprintf('Slow_amp_whole_%i.csv',i);
    name_time = sprintf('Whole_Time_%i.csv.csv',i);
    writematrix(slow_vars_interp_full,name_amp,'Delimiter',',');
    writematrix(t_tmp{1}(1):t_tmp{1}(end),name_time,'Delimiter',',');
    
    name_amp = sprintf('Slow_amp_pre_%i.csv',i);
    name_time = sprintf('Pre_Time_%i.csv.csv',i);
    writematrix(slow_vars_interp_pre,name_amp,'Delimiter',',');
    writematrix(t_tmp{2}(1):t_tmp{2}(end),name_time,'Delimiter',',');
    
    name_amp = sprintf('Slow_amp_post_%i.csv',i);
    name_time = sprintf('Post_Time_%i.csv.csv',i);
    writematrix(slow_vars_interp_post,name_amp,'Delimiter',',');
    writematrix(t_tmp{3}(1):t_tmp{3}(end),name_time,'Delimiter',',');
    
    name = sprintf('Omegas_%i.csv',i);
    writematrix(ws_MC,name,'Delimiter',',');
    
    hz_slowflow_full_truth=SF2fulltime(t_tmp{1},t_tmp{1},slow_vars{1},ws_MC,N_fmodes);
    hz_slowflow_pre_truth=SF2fulltime(t_tmp{2},t_tmp{2},slow_vars{2},ws_MC,N_fmodes);
    hz_slowflow_post_truth=SF2fulltime(t_tmp{3},t_tmp{3},slow_vars{3},ws_MC,N_fmodes);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%     figure;
%     f=gcf;
%     plot(t_full/(60 * samplerate), hz_slowflow_full_truth,'k','Linewidth',1)
%     hold on;
%     plot(t_full_plot/(60 * samplerate), h_full_plot,'r-','Linewidth',0.5)
%     
%     %plot(t_pre, hz_slowflow_pre_truth,'k-.','Linewidth',2)
%     plot(t_pre_plot/(60 * samplerate), h_pre_plot,'b-','Linewidth',0.5)
%     
%     %plot(t_post, hz_slowflow_post_truth,'k-.','Linewidth',2)
%     plot(t_post_plot/(60 * samplerate), h_post_plot,'m-','Linewidth',0.5)
%     xlabel('Time (in minutes)');
%     ylabel('Surface elevation (\eta)');
%     saveas(f,'Whole and Train Data Fits.tiff')
    
    figure;
    g = gcf;
    for j=1:length(slow_vars_interp_pre(:,1))
        %g = gcf;
        plot(slow_vars_interp_pre(j,:))
        hold on;
    end
    name2 = sprintf('Pre Slow Amps i=%d.tiff',i);
    
    g.PaperUnits = 'inches';
    g.PaperPosition = [0 0 15 5];
    print('-dtiff','-r600',name2)
    
%     figure;
%     %g = gcf;
%     for j=1:length(slow_vars_interp_full(:,1))
%         plot(slow_vars_interp_full(j,:))
%         hold on;
%     end
%     name2 = sprintf('Whole Slow Amps i=%d.tiff',i);
%     print('-dtiff','-r600',name2)
%     
%     figure;
%     g = gcf;
%     for j=1:length(slow_vars_interp_post(:,1))
%         plot(slow_vars_interp_post(j,:))
%         hold on;
%     end
%     name2 = sprintf('Post Slow Amps i=%d.tiff',i);
%     print('-dtiff','-r600',name2)
end