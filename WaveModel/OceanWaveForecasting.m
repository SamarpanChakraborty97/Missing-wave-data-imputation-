 %%

clear all
close all
clc

% load data set
% the utilized data set from San Nicholas island can be downloaded from
% http://thredds.cdip.ucsd.edu/thredds/catalog/cdip/archive/067p1/catalog.html?dataset=CDIP_Archive/067p1/067p1_d26.nc
% and more data sets are availiable at
% http://thredds.cdip.ucsd.edu/thredds/catalog/cdip/archive/catalog.html

src='067p1_d26.nc';

% Alternatively used data sets.
%src='Data_CDIP/Buoy 187 Maui/187p1_d06.nc';
%src='Data_CDIP/Buoy 217 Onslow Bay/217p1_d02.nc';

ncdisp(src)

info=ncinfo(src,'xyzZDisplacement');
Starttime=ncread(src,'xyzStartTime');
samplerate=ncread(src,'xyzSampleRate');
filt_delay=ncread(src,'xyzFilterDelay');
BeginDate=datetime(1970,1,1)+seconds(Starttime-filt_delay)
EndDate=datetime(1970,1,1)+seconds(Starttime-filt_delay)+seconds(info.Size/samplerate)

%%

eps=1*10^-3; % Set the parameter epsilon (cf. equation (1) and (2))
alpha = 0.8;
N_pts_per_wave=1;
N_windows=1; % Number of windows for comparison
N_MC=1000; % Number of Monte Carlo samples while solving the nonlinear function optimization for frequencies (4)
N_fmodes=16; % Number of elementary waves/ Fourier modes, i.e. parameter Nf in equation % (1)

% Initialize errors for comparison
err_const=NaN(N_windows,1);
err0=NaN(N_windows,1);
err_MC=NaN(N_windows,1);
cost_const=NaN(N_windows,1);
cost_SF=NaN(N_windows,1);
% Number of waves in the inital fitting interval. For the given
% observations 55 wave corresponds to approximately 3 min of observations.
N_waves=10;

% Number of waves after which the forecast is started. 255 waves yield
% approximately 15 minutes (cf. Fig. 3).
fit_window=20;
% Subtract time spend for fitting dynamical systems. Approximately two
% minutes (cf. Figure 3).
lag_steps= floor(1*60*samplerate);
% Forecasting horizon.
forecast_steps=floor(1*60*samplerate)+lag_steps;%

window_size=8* 10^3;

% Initialize array to store results and errors.
err_SF=zeros(N_windows,forecast_steps);
err_L2_SF=zeros(N_windows,forecast_steps);
err_triv_SF=zeros(N_windows,forecast_steps);
err_DMD_SF=zeros(N_windows,forecast_steps);
err_LSTM_SF=zeros(N_windows,forecast_steps);
err_mean_SF=zeros(N_windows,forecast_steps);


err_fit=zeros(N_windows,2);
err_L2=zeros(N_windows,2);
err_DMD=zeros(N_windows,2);
err_LSTM=zeros(N_windows,2);
err_triv=zeros(N_windows,2);
err_mean=zeros(N_windows,2);
err_gaus=zeros(N_windows,2);
err_h=zeros(N_windows,3);
err_t=zeros(N_windows,3);
err_begin=zeros(N_windows,1);

h_cmp_LSTM_1min=[];
h_cmp_DMD_1min=[];
h_cmp_LSQ_1min=[];
h_cmp_gauss_1min=[];
h_cmp_fit_1min=[];
h_cmp_1min=[];


h_cmp_LSTM_3min=[];
h_cmp_DMD_3min=[];
h_cmp_LSQ_3min=[];
h_cmp_gauss_3min=[];
h_cmp_fit_3min=[];
h_cmp_3min=[];

h_cmp_LSTM_5min=[];
h_cmp_DMD_5min=[];
h_cmp_LSQ_5min=[];
h_cmp_gauss_5min=[];
h_cmp_fit_5min=[];
h_cmp_5min=[];


% Iterate for each forecasting experiment.
for jj=1:N_windows
    % temporary variables
    h_tmp=cell(3,1);
    t_tmp=cell(3,1);
    hz_slowflow=cell(3,1);
    slow_vars=cell(3,1);
    % select a random start time
    w_start=randi(info.Size-window_size);
    
    % load z displacement from buoy data
    hz=ncread(src,'xyzZDisplacement',w_start,window_size).';
    % extract extreme values, i.e. trough depths and crests heights
    [tmp_wave_height ,tmp_wave_idx,zero_idx, ~]=my_wave_height_filter(hz ,N_pts_per_wave);
    ts=[tmp_wave_idx   ];%  tmp_t0];%[1:window_size];%
    hs=[tmp_wave_height ];% zeros(1,length(zero_idx))];%hz(jj,:);%
    
    [ts,id]=sort(ts);
    hs=hs(id);
    
    % Nonlinear function optimization (4). The parameter N_wave(1) controls
    % how many wavves are considered within this step (cf. Figure 3). 
    t1=ts(1:N_waves(1));
    h1=hs(1:N_waves(1));
    
    if N_fmodes<9
        % If less than nine Fourier modes are selected a MATLAB rutine can be
        % used for fitting.
        str_fmode=append('sin',num2str(N_fmodes));
        my_fit=fit(t1.',(h1-mean(h1)).',str_fmode);
        err_const(jj)=sum((h1-mean(h1)-my_fit(t1).').^2);
        coeff2=coeffvalues(my_fit);
        ws=sort(coeff2(2:3:end));
        tic
        % Monte Carlo sampling for optimization (4)
        [ws_MC, err_MC(jj)]= my_Freq_MC_fit(t1,h1,err_const(jj),N_fmodes,ws,N_MC,samplerate);
        toc
    else
        ws=zeros(1,N_fmodes);
        err_const(jj)=sum((h1-mean(h1)).^2);
        tic
        % Monte Carlo sampling for optimization (4)
        [ws_MC, err_MC(jj)]= my_Freq_MC_fit2(t1,h1,err_const(jj),N_fmodes,ws,N_MC,samplerate);
        toc
    end
    % Fitting the slowly varying amplitudes, i.e. solving the linear
    % optimization (5), for the first three minutes (cf. Fig. 3).
    tic
    [hz_slowflow{1} ,slow_vars{1}, ~]= my_SF_fit(t1, h1,ws_MC,eps,alpha);
    
    toc
    h_tmp{1}=h1;
    t_tmp{1}=t1;
    tic
    
    t_tmp{2}= ts(1:N_waves(1)+fit_window);
    h_tmp{2}= hs(1:N_waves(1)+fit_window);%[h_tmp(ii,2:end) hs(N_waves(1)+ii)];
    
    % Fitting the slowly varying amplitudes, i.e. solving the linear
    % optimization (5) for 15 minutes (cf. Fig. 3).
    [hz_slowflow{2} ,slow_vars{2},~]= my_SF_fit(t_tmp{2}, h_tmp{2},ws_MC,eps,alpha);
    
    window_idx= find(tmp_wave_idx> t_tmp{2}(end)+forecast_steps-lag_steps,1,'first');
    t_tmp{3}= ts(1:window_idx);
    h_tmp{3}= hs(1:window_idx);%[h_tmp(ii,2:end) hs(N_waves(1)+ii)];
    % Fitting the slowly varying amplitudes, i.e. solving the linear
    % optimization (5) for the whole time interval (cf. Fig. 3).
    [hz_slowflow{3} ,slow_vars{3},~]= my_SF_fit(t_tmp{3}, h_tmp{3},ws_MC,eps, alpha);
    
    toc
    jj
    % Calculate the number of slowly varyin amplitudes
    N_dim=2*N_fmodes+1;
    
    
    % Utilize the slowly varying amplitudes from the first 15 minutes to
    % fit the dynamical systems (cf. Fig. 3).
    slow_vars_fit=slow_vars{2};
    tt=t_tmp{2}(1):t_tmp{2}(end)-lag_steps;
    
    % Interpolation yielding uniform time series for the slowly varying amplitudes. 
    slow_vars_interp =interp1(t_tmp{2},slow_vars_fit.',tt,'spline','extrap').';
    
    %Normalize data
    m0= mean(slow_vars_interp,2);
    m1=std(slow_vars_interp,0,2);
    slow_vars_nomean=(slow_vars_interp-m0)./m1;
    
    %----------------
    % DMD Forecasting (cf. section 2.3.1)
    %----------------
    % Delay dimension M in equation (7)
    
%     delay_DMD=50;
%     
%     % Obtain least squares solution (10)
%     A_DMD=zeros(N_dim,delay_DMD*N_dim);
%     
%     % A seperate DMD model is fitted for each slowly varying amplitude.
%     % Alternatives have been tried, but were less successfull. 
%     for iter_dim=1:N_dim
%         zs_mat=[];
%         for dd=1:delay_DMD
%             zs_mat=[zs_mat; slow_vars_nomean(iter_dim,dd:end-delay_DMD+dd)];
%         end
%         V1=zs_mat(:,1:end-1);
%         V2=zs_mat(:,2:end);
%         [U,S,V]=svd(V1,'econ');
%         svds=1./diag(S);
%         svds((svds>1))=0;
%         A_tmp=V2*V*diag(svds)*U';
%         A_DMD(iter_dim,iter_dim:N_dim:end)=A_tmp(end,:);
%     end
%     % The matrix A_DMD advances the trajectory matrix by one step in time.
%     
%     
%     %----------------
%     % LSQ Forecasting (cf. section 2.3.1)
%     %----------------
%     delay_L2=100;
%     % Delay dimension M in equation (7)
%     A_L2=zeros(N_dim,delay_L2*N_dim);
%     
%     % A seperate LSQ model is fitted for each slowly varying amplitude.
%     % Alternatives have been tried, but were less successfull. 
%     for iter_dim=1:N_dim
%         if m1(iter_dim)>0
%             zs_mat=[];
%             for dd=1:delay_L2
%                 zs_mat=[zs_mat; slow_vars_nomean(iter_dim,dd:end-delay_L2+dd-1)];
%             end
%             
%             ys_vec=slow_vars_nomean(iter_dim,delay_L2+1:end).';%-slow_vars_nomean(jj,delay_dim:end-1).';%ys_delta(jj,:).';%
%             A_vec=zs_mat.'\ys_vec;
%             A_L2(iter_dim,iter_dim:N_dim:end)=A_vec;
%         else
%             A_L2(iter_dim,end-N_dim+iter_dim)=1;
%         end
%         
%         
%     end
%     %----------------
%     % LSTM Network (cf. section 2.3.2)
%     %----------------
%     % Contrary to the linear approaches a single network simultaneously
%     % forecasting all slowly varying amplitudes in constructed. 
%     numResponses=   N_dim;
%     numFeatures=   N_dim;
%     % Number of LSTM cells (hidden units)
%     numHiddenUnits=200;
%     
%     % Constructing LSTM network
%     layers = [ ...
%         sequenceInputLayer(numFeatures)
%         lstmLayer(numHiddenUnits,'OutputMode','sequence')
%         fullyConnectedLayer(100)
%         dropoutLayer(0.01)
%         fullyConnectedLayer(numResponses)
%         regressionLayer];
%     
%     % Setting training options
%     options = trainingOptions('adam', ...
%         'MaxEpochs',150, ...
%         'GradientThreshold',1, ...
%         'InitialLearnRate',0.005, ...
%         'LearnRateSchedule','piecewise', ...
%         'LearnRateDropPeriod',125, ...
%         'LearnRateDropFactor',0.2, ...
%         'Verbose',0);
%     % Train neural network
%     net    = trainNetwork(slow_vars_nomean(:,1:end-1) ,slow_vars_nomean(:,2:end) ,layers,options);
%     
%     
%     %----------------
%     % Forecasting 
%     %----------------
%     
%     
%     % Initializing variables
%     N_t=length(slow_vars_nomean(1,:));
%     slow_vars_forecast_L2=zeros(N_dim,forecast_steps+N_t);
%     slow_vars_forecast_triv=zeros(N_dim,forecast_steps+N_t);
%     slow_vars_forecast_DMD=zeros(N_dim,forecast_steps+N_t);
%     slow_vars_forecast_LSTM=zeros(N_dim,forecast_steps+N_t);
%     
%     
%     slow_vars_forecast_L2(:,1:length(slow_vars_nomean(1,:)))=slow_vars_nomean;
%     slow_vars_forecast_triv(:,1:length(slow_vars_nomean(1,:)))=slow_vars_nomean;
%     slow_vars_forecast_DMD(:,1:length(slow_vars_nomean(1,:)))=slow_vars_nomean;
%     slow_vars_forecast_LSTM(:,1:length(slow_vars_nomean(1,:)))=slow_vars_nomean;
%     
%     % The three dynamical system (LSTM, DMD, LSQ) are iterratively applied
%     % to data (cf. equation (6)) yielding forecasts of, at least in
%     % principle, arbitrary length. 
%     
%     for iter_forecast=1:forecast_steps
%         
%         tmp=slow_vars_forecast_L2(:,N_t+iter_forecast+(-delay_L2:-1));
%         slow_vars_forecast_L2(:,N_t+iter_forecast)=A_L2*tmp(:);%slow_vars_forecast_own(:,jj+delay_dim-1)+
%         
%         tmp=slow_vars_forecast_DMD(:,N_t+iter_forecast+(-delay_DMD:-1));
%         slow_vars_forecast_DMD(:,N_t+iter_forecast)=A_DMD*tmp(:);%slow_vars_forecast_own(:,jj+delay_dim-1)+
%         
%         
%         slow_vars_forecast_triv(:,N_t+iter_forecast)=slow_vars_nomean(:,end);
%         %   pred=zeros(N_dim,N_t+iter_forecast-1);
%         
%         %   for iter_dim=1:N_dim
%         %  [nets{ iter_dim},pred( iter_dim,:)]=predictAndUpdateState(nets{iter_dim},slow_vars_forecast_LSTM( iter_dim,1:N_t+iter_forecast-1),'MiniBatchSize',1);
%         %   end
%         [net, pred] =predictAndUpdateState(net ,slow_vars_forecast_LSTM(:,1:N_t+iter_forecast-1),'MiniBatchSize',1);
%         slow_vars_forecast_LSTM(:,N_t+iter_forecast)= pred(:,end) ;
%         
%         
%     end
%     
%     % Rescaling data to original distribution
%     slow_vars_forecast_DMD=m0+slow_vars_forecast_DMD.*m1;
%     slow_vars_forecast_L2=m0+slow_vars_forecast_L2.*m1;
%     slow_vars_forecast_triv=m0+slow_vars_forecast_triv.*m1;
%     slow_vars_forecast_LSTM=m0+slow_vars_forecast_LSTM.*m1;
%     
%     % Deleting the first steps, which are not forecasts. 
%     slow_vars_forecast_DMD(:,1:N_t)=[];
%     slow_vars_forecast_L2(:,1:N_t)=[];
%     slow_vars_forecast_triv(:,1:N_t)=[];
%     slow_vars_forecast_LSTM(:,1:N_t)=[];
%     
%     slow_vars_forecast_mean=repmat(m0,1,forecast_steps);
%     
%    ,2 slow_vars_truth=zeros(N_dim,forecast_steps);
%     for iter_forecast=1:forecast_steps
%        t_idx=t_tmp{2}(end)+iter_forecast-lag_steps;
%        % Obtaining the best fit for the slow flow at each forcasting step 
%        slow_vars_truth(:,1:iter_forecast)=interp1(t_tmp{3},slow_vars{3}.',(t_tmp{2}(end)-lag_steps+1:t_idx),'spline','extrap').';
%         
%         % Computing the L2 error of the slow flow
%         err_SF(jj,iter_forecast)=mean(vecnorm(slow_vars_truth(:,1:iter_forecast)).^2);
%         err_L2_SF(jj,iter_forecast)=mean(vecnorm(slow_vars_forecast_L2(:,1:iter_forecast)-slow_vars_truth(:,1:iter_forecast)).^2);
%         err_DMD_SF(jj,iter_forecast)=mean(vecnorm(slow_vars_forecast_DMD(:,1:iter_forecast)-slow_vars_truth(:,1:iter_forecast)).^2);
%         err_LSTM_SF(jj,iter_forecast)=mean(vecnorm(slow_vars_forecast_LSTM(:,1:iter_forecast)-slow_vars_truth(:,1:iter_forecast)).^2);
%         err_triv_SF(jj,iter_forecast)=mean(vecnorm(slow_vars_forecast_triv(:,1:iter_forecast)-slow_vars_truth(:,1:iter_forecast)).^2);
%         err_mean_SF(jj,iter_forecast)=mean(vecnorm(slow_vars_forecast_mean(:,1:iter_forecast)-slow_vars_truth(:,1:iter_forecast)).^2);
%         
%         
%     end
%     
%     % Converting the slowly varying amplitudes into a single surface
%     % elevation profile. 
%     hz_slowflow_full_truth=SF2fulltime(t_tmp{2}(end)-lag_steps+(1:forecast_steps ),t_tmp{3},slow_vars{3},ws_MC,N_fmodes);
%     hz_slowflow_full_L2=SF2fulltime(t_tmp{2}(end)-lag_steps+(1:forecast_steps ),t_tmp{2}(end)-lag_steps+(1:forecast_steps),slow_vars_forecast_L2,ws_MC,N_fmodes);
%     hz_slowflow_full_DMD=SF2fulltime(t_tmp{2}(end)-lag_steps+(1:forecast_steps ),t_tmp{2}(end)-lag_steps+(1:forecast_steps),slow_vars_forecast_DMD,ws_MC,N_fmodes);
%     hz_slowflow_full_LSTM=SF2fulltime(t_tmp{2}(end)-lag_steps+(1:forecast_steps ),t_tmp{2}(end)-lag_steps+(1:forecast_steps),slow_vars_forecast_LSTM,ws_MC,N_fmodes);
%     hz_slowflow_full_triv=SF2fulltime(t_tmp{2}(end)-lag_steps+(1:forecast_steps ),t_tmp{2}(end)-lag_steps+(1:forecast_steps),slow_vars_forecast_triv,ws_MC,N_fmodes);
%     hz_slowflow_full_mean=SF2fulltime(t_tmp{2}(end)-lag_steps+(1:forecast_steps ),t_tmp{2}(end)-lag_steps+(1:forecast_steps),slow_vars_forecast_mean,ws_MC,N_fmodes);
%     
%     % Deducting the crest heights and trough depths from the individual
%     % surface elevation profiles. 
%     [fit_wave_height ,fit_time,~, ~]=my_wave_height_filter(hz_slowflow_full_truth ,N_pts_per_wave);
%     [L2_wave_height ,L2_forecast_time,~, ~]=my_wave_height_filter(hz_slowflow_full_L2 ,N_pts_per_wave);
%     [DMD_wave_height ,DMD_forecast_time,~, ~]=my_wave_height_filter(hz_slowflow_full_DMD ,N_pts_per_wave);
%     [LSTM_wave_height ,LSTM_forecast_time,~, ~]=my_wave_height_filter(hz_slowflow_full_LSTM ,N_pts_per_wave);
%     [triv_wave_height ,triv_forecast_time,~, ~]=my_wave_height_filter(hz_slowflow_full_triv ,N_pts_per_wave);
%     [mean_wave_height ,mean_forecast_time,~, ~]=my_wave_height_filter(hz_slowflow_full_mean ,N_pts_per_wave);
%     
%     % Finding the index at which the real time forecasts are obtained. 
%     idx_start=find(t_tmp{3}>(t_tmp{2}(end)),1,'first');
%     
%     % Computing mean absolute error
%     err0(jj)=mean(abs([h_tmp{3}(idx_start:end-1)]))%;0.*t_tmp{3}(idx_start:end-1)%
%     
%     
%     fit_time=fit_time+t_tmp{2}(end)-lag_steps;
%     % Extracting fitted wave heights
%     [h_cmp_fit, t_cmp_fit]=my_wave_height_compare(h_tmp{3}(idx_start:end-1),t_tmp{3}(idx_start:end-1),fit_wave_height,fit_time);
%     % Computing mean absolute error of fitted wave heights
%     err_fit(jj,:)=[mean(abs(h_tmp{3}(idx_start:end-1) -h_cmp_fit)) mean(abs( t_tmp{3}(idx_start:end-1)- t_cmp_fit))];
%     
%     
%     L2_forecast_time=L2_forecast_time+t_tmp{2}(end)-lag_steps;
%     % Extracting wave heights from LSQ forecast
%     [h_cmp_L2,t_cmp_L2] =my_wave_height_compare(h_tmp{3}(idx_start:end-1),t_tmp{3}(idx_start:end-1),L2_wave_height,L2_forecast_time);
%     % Computing mean absolute error of wave heights from LSQ forecast
%     err_L2(jj,:)=[mean(abs(h_tmp{3}(idx_start:end-1) -h_cmp_L2)) mean(abs(t_tmp{3}(idx_start:end-1) -t_cmp_L2))];
%     
%     DMD_forecast_time=DMD_forecast_time+t_tmp{2}(end)-lag_steps;
%     % Extracting wave heights from DMD forecast
%     [h_cmp_DMD, t_cmp_DMD]=my_wave_height_compare(h_tmp{3}(idx_start:end-1),t_tmp{3}(idx_start:end-1),DMD_wave_height,DMD_forecast_time);
%     % Computing mean absolute error of wave heights from DMD forecast
%     err_DMD(jj,:)=[mean(abs(h_tmp{3}(idx_start:end-1) -h_cmp_DMD)) mean(abs(t_tmp{3}(idx_start:end-1) -t_cmp_DMD))];
%     
%     LSTM_forecast_time=LSTM_forecast_time+t_tmp{2}(end)-lag_steps;
%     % Extracting wave heights from LSTM forecast
%     [h_cmp_LSTM, t_cmp_LSTM]=my_wave_height_compare(h_tmp{3}(idx_start:end-1),t_tmp{3}(idx_start:end-1),LSTM_wave_height,LSTM_forecast_time);
%     % Computing mean absolute error of wave heights from LSTM forecast
%     err_LSTM(jj,:)=[mean(abs(h_tmp{3}(idx_start:end-1) -h_cmp_LSTM)) mean(abs(t_tmp{3}(idx_start:end-1) -t_cmp_LSTM))];
%     
%     
%     triv_forecast_time=triv_forecast_time+t_tmp{2}(end)-lag_steps;
%     % Extracting wave heights from trivial forecast 
%     [h_cmp_triv, t_cmp_triv]=my_wave_height_compare(h_tmp{3}(idx_start:end-1),t_tmp{3}(idx_start:end-1),triv_wave_height,triv_forecast_time);
%     % Computing mean absolute error of wave heights from trivial forecast
%     err_triv(jj,:)=[mean(abs(h_tmp{3}(idx_start:end-1)-h_cmp_triv)) mean(abs(t_tmp{3}(idx_start:end-1)-t_cmp_triv))];
%     
%     mean_forecast_time=mean_forecast_time+t_tmp{2}(end)-lag_steps;
%     % Extracting wave heights from mean forecast
%     [h_cmp_mean,t_cmp_mean]=my_wave_height_compare(h_tmp{3}(idx_start:end-1),t_tmp{3}(idx_start:end-1),mean_wave_height,mean_forecast_time);
%     % Computing mean absolute error of wave heights from mean forecast
%     err_mean(jj,:)=[ mean(abs(h_tmp{3}(idx_start:end-1) -h_cmp_mean)) mean(abs(t_tmp{3}(idx_start:end-1) -t_cmp_mean))];
%     
%     %--------------------------
%     % Constructing the benchmark of a single wave train
%     %--------------------------
%     
%     % Half of the mean wave period
%     T_mean=mean(diff(t_tmp{3}(1:idx_start-1)));
%     % Number of waves in the forecast
%     N_waves_forecast=floor(( t_tmp{3}(end)-t_tmp{3}(idx_start))/T_mean)+1;
%     %mean_guess=[repmat(mean(abs(h_tmp{2})),1,N_waves_forecast);t_tmp{3}(idx_start)+T_mean.* (0:N_waves_forecast-1)];
%     
%     % Mean wave height
%     mean_wave_guess=repmat(mean(abs(h_tmp{2})),1,N_waves_forecast);
%     %mean_wave_sign=ones(1,N_waves_forecast);
%     % If first value is positive the benchmark starts with a positive guess
%     % and negtive if otherwise. The other values alternate. ( i.e. crest,
%     % through, crest, trough,... or trough, crest,trough, crest,...)
%     if h_tmp{3}(idx_start-1)<0
%         mean_wave_guess(2:2:end)=-mean_wave_guess(2:2:end);
%     else
%         mean_wave_guess(1:2:end)=-mean_wave_guess(1:2:end);
%     end
%     % Extracting wave heights from benchmark
%     [h_cmp_gauss, t_cmp_gauss]=my_wave_height_compare(h_tmp{3}(idx_start:end-1),t_tmp{3}(idx_start:end-1),mean_wave_guess,t_tmp{3}(idx_start)+T_mean.* (0:N_waves_forecast-1));
%     
%     
%     % Computing mean absolute error of wave heights from benchmark
%     err_gaus(jj,:)=[mean(abs(h_tmp{3}(idx_start:end-1)-h_cmp_gauss))  mean(abs(t_tmp{3}(idx_start:end-1)-t_cmp_gauss))];
%     
%     
%     % Extracting the largest crest height/trough depths (for Fig. 13 b, 14 b and 14d)
%     [max_wave_L2,max_wave_L2_idx]=max(abs(hz_slowflow_full_L2));
%     [max_wave_DMD,max_wave_DMD_idx]=max(abs(hz_slowflow_full_DMD));
%     [max_wave_LSTM,max_wave_LSTM_idx]=max(abs(hz_slowflow_full_LSTM));
%     
%     
%     [max_wave_truth,max_wave_truth_idx]=max(abs(h_tmp{3}(idx_start:end-1)));
%     
%     % Computing the error of the largest crest/trough depths 
%     err_h(jj,1)=abs(max_wave_L2-max_wave_truth);
%     err_t(jj,1)=abs(max_wave_L2_idx-t_tmp{3}(max_wave_truth_idx));
%     
%     err_h(jj,2)=abs(max_wave_DMD-max_wave_truth);
%     err_t(jj,2)=abs(max_wave_DMD_idx-t_tmp{3}(max_wave_truth_idx));
%     
%     err_h(jj,3)=abs(max_wave_LSTM-max_wave_truth);
%     err_t(jj,3)=abs(max_wave_LSTM_idx-t_tmp{3}(max_wave_truth_idx));
%     
%     ts=(t_tmp{3}(idx_start:end-1)-t_tmp{3}(idx_start))/samplerate;
%     
%     idx_1min=find(ts>60,1)-1;
%     idx_3min=find(ts>3*60,1)-1;
%     
%     % Saving the individual forecasted crest heights and trough depths for 1 minute
%     h_cmp_LSTM_1min=[h_cmp_LSTM_1min h_cmp_LSTM(1:idx_1min) ];
%     h_cmp_DMD_1min=[h_cmp_DMD_1min h_cmp_DMD(1:idx_1min)];
%     h_cmp_LSQ_1min=[h_cmp_LSQ_1min  h_cmp_L2(1:idx_1min)];
%     h_cmp_gauss_1min=[h_cmp_gauss_1min h_cmp_gauss(1:idx_1min)];
%     h_cmp_fit_1min=[h_cmp_fit_1min h_cmp_fit(1:idx_1min)];
%     h_cmp_1min=[h_cmp_1min h_tmp{3}(idx_start:idx_start+idx_1min-1)];
%     
%     % Saving the individual forecasted crest heights and trough depths for
%     % 3 minutes
%     h_cmp_LSTM_3min=[h_cmp_LSTM_3min h_cmp_LSTM(1:idx_3min) ];
%     h_cmp_DMD_3min=[h_cmp_DMD_3min h_cmp_DMD(1:idx_3min)];
%     h_cmp_LSQ_3min=[h_cmp_LSQ_3min  h_cmp_L2(1:idx_3min)];
%     h_cmp_gauss_3min=[h_cmp_gauss_3min h_cmp_gauss(1:idx_3min)];
%     h_cmp_fit_3min=[h_cmp_fit_3min h_cmp_fit(1:idx_3min)];
%     h_cmp_3min=[h_cmp_3min h_tmp{3}(idx_start:idx_start+idx_3min-1)];
%     
%     % Saving the individual forecasted crest heights and trough depths for
%     % 5 minutes
%     h_cmp_LSTM_5min=[h_cmp_LSTM_5min h_cmp_LSTM ];
%     h_cmp_DMD_5min=[h_cmp_DMD_5min h_cmp_DMD ];
%     h_cmp_LSQ_5min=[h_cmp_LSQ_5min  h_cmp_L2 ];
%     h_cmp_gauss_5min=[h_cmp_gauss_5min h_cmp_gauss ];
%     h_cmp_fit_5min=[h_cmp_fit_5min h_cmp_fit ];
%     h_cmp_5min=[h_cmp_5min h_tmp{3}(idx_start:end-1)];
       
end

figure;
f=gcf;
for i=1:length(slow_vars_nomean(:,2))
    plot(slow_vars_nomean(i,:))
    hold on;
end
saveas(f,'Amps.tiff')
    
%%

%---------------
% Plot results
%---------------
% Bar plots of the mean absolute error for all forecasted wave heights (cf.
% Fig. 13a, 14a and 14c).
err_plt=zeros(5,2);
err_std=zeros(5,2);
err_plt(1,:)=mean(err_L2,'omitnan' );
err_plt(2,:)=mean(err_DMD,'omitnan' );
err_plt(3,:)=mean(err_LSTM,'omitnan' );

err_std(1,:)=std(err_L2,'omitnan' );
err_std(2,:)=std(err_DMD,'omitnan' );
err_std(3,:)=std(err_LSTM,'omitnan' );

%err_plt(4,:)=mean(err_triv,'omitnan' );
%err_plt(5,:)=mean(err_mean,'omitnan' );
mean(err0)


err_plt(4,:)=mean(err_gaus,'omitnan' );
err_plt(5,:)=mean(err_fit,'omitnan' );

err_std(4,:)=std(err_gaus,'omitnan' );
err_std(5,:)=std(err_fit,'omitnan' );

X = categorical({'LSQ','DMD','LSTM','Benchmark','Fit'});
X = reordercats(X,{'LSQ','DMD','LSTM','Benchmark','Fit'});

figure
subplot(1,2,1)
bar(X,err_plt(:,1))
hold on
er=errorbar(X,err_plt(:,1),-err_std(:,1),err_std(:,1));
er.Color = [0 0 0];
er.LineStyle = 'none';
ylabel('average error in m')
subplot(1,2,2)
bar(X,err_plt(:,2)./samplerate)
hold on
er=errorbar(X,err_plt(:,2)./samplerate,(-err_std(:,2))./samplerate,(err_std(:,2))./samplerate);
er.Color = [0 0 0];
er.LineStyle = 'none';
ylabel('average error in sec')
%%

% Plot mean and variance of the difference between of the forecasted slow
% flow and the truth (=error). 
% Visualization not used in Publication. 


figure
%subplot(1,2,1)
hold on


times=(1:forecast_steps)./samplerate./60;

plot(times,mean(err_L2_SF))
hold on
plot(times,mean(err_DMD_SF))
plot(times,mean(err_LSTM_SF))
plot(times,mean(err_mean_SF))
plot(times,mean(err_triv_SF))

%pl(1)=my_uncertain_plt(times,mean(err_L2_SF./err_mean_SF),std(err_L2_SF./err_mean_SF));

%pl(2)=my_uncertain_plt(times,mean(err_DMD_SF./err_mean_SF),std(err_DMD_SF./err_mean_SF));

%pl(3)=my_uncertain_plt(times,mean(err_LSTM_SF./err_mean_SF),std(err_LSTM_SF./err_mean_SF));

lg=legend('LSQ','DMD','LSTM','Training mean','Last sample')
axis tight
plot(lag_steps/samplerate/60.*[1 1],get(gca,'YLim'),'--k')
%plot(get(gca,'XLim'),[1 1],'Color',[0.5 0.5 0.5])
xlabel('time (min)')
ylabel('relative mean square error')
%lg=legend(pl, 'L2-fit','DMD','LSTM');
set(lg,'Location','Southeast')
title('Mean')

figure
plot(times,std(err_L2_SF))
hold on
plot(times,std(err_DMD_SF))
plot(times,std(err_LSTM_SF))
plot(times,std(err_mean_SF))
plot(times,std(err_triv_SF))

lg=legend('LSQ','DMD','LSTM','Training mean','Last sample')
plot(lag_steps/samplerate/60.*[1 1],get(gca,'YLim'),'--k')
%plot(get(gca,'XLim'),[1 1],'Color',[0.5 0.5 0.5])

xlabel('time (min)')
ylabel('relative mean square error')
%lg=legend(pl, 'L2-fit','DMD','LSTM');
%set(lg,'Location','Northwest')
title('Variance')

%%
% Bar plots of the mean absolute error for the heighst wave (cf.
% Fig. 13b, 14b and 14d).
err_plt=mean(err_h,'omitnan' );
err_std=std(err_h,'omitnan' );

X = categorical({'LSQ','DMD','LSTM'});
X = reordercats(X,{'LSQ','DMD','LSTM'});

figure
bar(X,err_plt)
hold on
text(1:length(err_plt),err_plt+0.01,[num2str(round(err_plt,2)') ['\pm'; '\pm'; '\pm'] num2str(round(err_std',2)) [' m';' m';' m']],'vert','bottom','horiz','center');

ylabel('average maximal height/depths error (m)')
% ylim([0 0.65])
%%
% Scatter plots of the various forecasts  (cf. Fig. 12)
my_scatter(h_cmp_1min.',h_cmp_LSTM_1min.','LSTM network after one minute')
my_scatter(h_cmp_1min.',h_cmp_DMD_1min.','DMD after one minute')
my_scatter(h_cmp_1min.',h_cmp_LSQ_1min.','LSQ after one minute')
my_scatter(h_cmp_1min.',h_cmp_gauss_1min.','Benchmark after one minute')


my_scatter(h_cmp_3min.',h_cmp_LSTM_3min.','LSTM network after three minute')
my_scatter(h_cmp_3min.',h_cmp_DMD_3min.','DMD after three minute')
my_scatter(h_cmp_3min.',h_cmp_LSQ_3min.','LSQ after three minute')
my_scatter(h_cmp_3min.',h_cmp_gauss_3min.','Benchmark after three minute')

my_scatter(h_cmp_5min.',h_cmp_LSTM_5min.','LSTM network after five minute')
my_scatter(h_cmp_5min.',h_cmp_DMD_5min.','DMD after five minute')
my_scatter(h_cmp_5min.',h_cmp_LSQ_5min.','LSQ after five minute')
my_scatter(h_cmp_5min.',h_cmp_gauss_5min.','Benchmark after five minute')

% Construct Err valiable with fields for saving

Err.DMD.MAE.one_min=mean(abs(h_cmp_1min.'-h_cmp_DMD_1min.'));
Err.DMD.MAE.three_min=mean(abs(h_cmp_3min.'-h_cmp_DMD_3min.'));
Err.DMD.MAE.five_min=mean(abs(h_cmp_5min.'-h_cmp_DMD_5min.'));

Err.LSQ.MAE.one_min=mean(abs(h_cmp_1min.'-h_cmp_LSQ_1min.'));
Err.LSQ.MAE.three_min=mean(abs(h_cmp_3min.'-h_cmp_LSQ_3min.'));
Err.LSQ.MAE.five_min=mean(abs(h_cmp_5min.'-h_cmp_LSQ_5min.'));

Err.LSTM.MAE.one_min=mean(abs(h_cmp_1min.'-h_cmp_LSTM_1min.'));
Err.LSTM.MAE.three_min=mean(abs(h_cmp_3min.'-h_cmp_LSTM_3min.'));
Err.LSTM.MAE.five_min=mean(abs(h_cmp_5min.'-h_cmp_LSTM_5min.'));

Err.Ben.MAE.one_min=mean(abs(h_cmp_1min.'-h_cmp_gauss_1min.'));
Err.Ben.MAE.three_min=mean(abs(h_cmp_3min.'-h_cmp_gauss_3min.'));
Err.Ben.MAE.five_min=mean(abs(h_cmp_5min.'-h_cmp_gauss_5min.'));

Err.fit.MAE.one_min=mean(abs(h_cmp_1min.'-h_cmp_fit_1min.'));
Err.fit.MAE.three_min=mean(abs(h_cmp_3min.'-h_cmp_fit_3min.'));
Err.fit.MAE.five_min=mean(abs(h_cmp_5min.'-h_cmp_fit_5min.'));

Err.DMD.corr.one_min=corr(h_cmp_1min.',h_cmp_DMD_1min.');
Err.DMD.corr.three_min=corr(h_cmp_3min.',h_cmp_DMD_3min.');
Err.DMD.corr.five_min=corr(h_cmp_5min.',h_cmp_DMD_5min.');

Err.LSQ.corr.one_min=corr(h_cmp_1min.',h_cmp_LSQ_1min.');
Err.LSQ.corr.three_min=corr(h_cmp_3min.',h_cmp_LSQ_3min.');
Err.LSQ.corr.five_min=corr(h_cmp_5min.',h_cmp_LSQ_5min.');

Err.LSTM.corr.one_min=corr(h_cmp_1min.',h_cmp_LSTM_1min.');
Err.LSTM.corr.three_min=corr(h_cmp_3min.',h_cmp_LSTM_3min.');
Err.LSTM.corr.five_min=corr(h_cmp_5min.',h_cmp_LSTM_5min.');

Err.Ben.corr.one_min=corr(h_cmp_1min.',h_cmp_gauss_1min.');
Err.Ben.corr.three_min=corr(h_cmp_3min.',h_cmp_gauss_3min.');
Err.Ben.corr.five_min=corr(h_cmp_5min.',h_cmp_gauss_5min.');

Err.fit.corr.one_min=corr(h_cmp_1min.',h_cmp_fit_1min.');
Err.fit.corr.three_min=corr(h_cmp_3min.',h_cmp_fit_3min.');
Err.fit.corr.five_min=corr(h_cmp_5min.',h_cmp_fit_5min.');


Err.DMD.RSME.one_min=sqrt(mean((h_cmp_1min.'-h_cmp_DMD_1min.').^2));
Err.DMD.RSME.three_min=sqrt(mean((h_cmp_3min.'-h_cmp_DMD_3min.').^2));
Err.DMD.RSME.five_min=sqrt(mean((h_cmp_5min.'-h_cmp_DMD_5min.').^2));

Err.LSQ.RSME.one_min=sqrt(mean((h_cmp_1min.'-h_cmp_LSQ_1min.').^2));
Err.LSQ.RSME.three_min=sqrt(mean((h_cmp_3min.'-h_cmp_LSQ_3min.').^2));
Err.LSQ.RSME.five_min=sqrt(mean((h_cmp_5min.'-h_cmp_LSQ_5min.').^2));

Err.LSTM.RSME.one_min=sqrt(mean((h_cmp_1min.'-h_cmp_LSTM_1min.').^2));
Err.LSTM.RSME.three_min=sqrt(mean((h_cmp_3min.'-h_cmp_LSTM_3min.').^2));
Err.LSTM.RSME.five_min=sqrt(mean((h_cmp_5min.'-h_cmp_LSTM_5min.').^2));

Err.Ben.RSME.one_min=sqrt(mean((h_cmp_1min.'-h_cmp_gauss_1min.').^2));
Err.Ben.RSME.three_min=sqrt(mean((h_cmp_3min.'-h_cmp_gauss_3min.').^2));
Err.Ben.RSME.five_min=sqrt(mean((h_cmp_5min.'-h_cmp_gauss_5min.').^2));

Err.fit.RSME.one_min=sqrt(mean((h_cmp_1min.'-h_cmp_fit_1min.').^2));
Err.fit.RSME.three_min=sqrt(mean((h_cmp_3min.'-h_cmp_fit_3min.').^2));
Err.fit.RSME.five_min=sqrt(mean((h_cmp_5min.'-h_cmp_fit_5min.').^2));

%%