clc
clear all

mins = 1;
threshold = 1.0;

lstm = readmatrix('LSTM_errors.txt');
t_lstm = readmatrix('t_LSTM_errors.txt');
idx = find(abs(lstm(:,1))>threshold);
lstm(idx,:) = NaN;
t_lstm(idx,:) = NaN;

cnn_lstm_fit = readmatrix('CNN_LSTM_Fit_errors.txt');
t_cnn_lstm_fit = readmatrix('t_CNN_LSTM_Fit_errors.txt');
idx = find(abs(cnn_lstm_fit(:,1))>threshold);
cnn_lstm_fit(idx,:) = NaN;
t_cnn_lstm_fit(idx,:) = NaN;

cnn_lstm_amps = readmatrix('CNN_LSTM_Slow_errors.txt');
t_cnn_lstm_amps = readmatrix('t_CNN_LSTM_Slow_errors.txt');
idx = find(abs(cnn_lstm_amps(:,1))>threshold);
cnn_lstm_amps(idx,:) = NaN;
t_cnn_lstm_amps(idx,:) = NaN;

ssa = readmatrix('SSA_errors.txt');
t_ssa = readmatrix('t_SSA_errors.txt');
idx = find(abs(ssa(:,1))>threshold);
ssa(idx,:) = NaN;
t_ssa(idx,:) = NaN;

fit = readmatrix('Fit_errors.txt');
t_fit = readmatrix('t_Fit_errors.txt');
idx = find(abs(fit(:,1))>threshold);
fit(idx,:) = NaN;
t_fit(idx,:) = NaN;

baseline = readmatrix('Baseline_errors.txt');
t_baseline = readmatrix('t_Baseline_errors.txt');
idx = find(abs(baseline(:,1))>threshold);
baseline(idx,:) = NaN;
t_baseline(idx,:) = NaN;

% lstm_max = readmatrix('LSTM_max_errors.txt');
% cnn_lstm_fit_max = readmatrix('CNN_LSTM_Fit_max_errors.txt');
% cnn_lstm_amps_max = readmatrix('CNN_LSTM_Slow_max_errors.txt');
% ssa_max = readmatrix('SSA_max_errors.txt');
% fit_max = readmatrix('Fit_max_errors.txt');
% baseline_max = readmatrix('Baseline_max_errors.txt');

lstm_cum = readmatrix('LSTM_cumulative.txt');
cnn_lstm_fit_cum = readmatrix('CNN_LSTM_Fit_cumulative.txt');
cnn_lstm_amps_cum = readmatrix('CNN_LSTM_Slow_cumulative.txt');
ssa_cum = readmatrix('SSA_cumulative.txt');
fit_cum = readmatrix('Fit_cumulative.txt');
baseline_cum = readmatrix('Baseline_cumulative.txt');
truth_cum = readmatrix('Truth_cumulative.txt');

windows = length(fit(:,1));
arrs = [lstm(:,1) cnn_lstm_fit(:,1)  cnn_lstm_amps(:,1) ssa(:,1)  fit(:,1)  baseline(:,1)]; 
arrs_time = [t_lstm(:,1)  t_cnn_lstm_fit(:,1)  t_cnn_lstm_amps(:,1) t_ssa(:,1)  t_fit(:,1)  t_baseline(:,1)];
num_arrs = length(arrs(1,:));
plotBarCats(arrs, arrs_time, num_arrs, mins, windows, threshold)

% arrs = [lstm_max(:,1) cnn_lstm_fit_max(:,1)  cnn_lstm_amps_max(:,1) ssa_max(:,1)  fit_max(:,1)  baseline_max(:,1)];
% num_arrs = length(arrs(1,:));
% plotBarMax(arrs, num_arrs, mins, windows)

idx = find(isnan(lstm_cum));
idx2 = find(abs(lstm_cum)>2.0);
idx = horzcat(idx, idx2);
lstm_cum(idx) = [];
truth_cum_lstm = truth_cum;
truth_cum_lstm(idx) = [];
my_scatter2(truth_cum_lstm.',lstm_cum.','LSTM and Truth comparisons',mins, windows, threshold)

idx = find(isnan(cnn_lstm_fit_cum));
idx2 = find(abs(cnn_lstm_fit_cum)>2.0);
idx = horzcat(idx, idx2);
cnn_lstm_fit_cum(idx) = [];
truth_cum_cnn_fit_lstm = truth_cum;
truth_cum_cnn_fit_lstm(idx) = [];
my_scatter2(truth_cum_cnn_fit_lstm.',cnn_lstm_fit_cum.','CNN+LSTM(fit) and Truth comparisons',mins, windows, threshold)

idx = find(isnan(cnn_lstm_amps_cum));
idx2 = find(abs(cnn_lstm_amps_cum)>2.0);
idx = horzcat(idx, idx2);
cnn_lstm_amps_cum(idx) = [];
truth_cum_cnn_amps_lstm = truth_cum;
truth_cum_cnn_amps_lstm(idx) = [];
my_scatter2(truth_cum_cnn_amps_lstm.',cnn_lstm_amps_cum.','CNN+LSTM(amplitudes) and Truth comparisons',mins, windows, threshold)

idx = find(isnan(fit_cum));
idx2 = find(abs(fit_cum)>2.0);
idx = horzcat(idx, idx2);
fit_cum(idx) = [];
truth_cum_fit = truth_cum;
truth_cum_fit(idx) = [];
my_scatter2(truth_cum_fit.',fit_cum.','Fit and Truth comparisons',mins, windows, threshold)

idx = find(isnan(ssa_cum));
idx2 = find(abs(ssa_cum)>2.0);
idx = horzcat(idx, idx2);
ssa_cum(idx) = [];
truth_cum_ssa = truth_cum;
truth_cum_ssa(idx) = [];
my_scatter2(truth_cum_ssa.',ssa_cum.','SSA and Truth comparisons',mins, windows, threshold)

%idx = find(isnan(baseline_cum));
%idx2 = find(abs(baseline_cum)>threshold);
%idx = horzcat(idx, idx2);
%baseline_cum(idx) = [];
%truth_cum_baseline = truth_cum;
%truth_cum_baseline(idx) = [];
%my_scatter2(truth_cum_baseline.',baseline_cum.','Baseline and Truth comparisons',mins, windows, threshold);

arrs_mae = [lstm(:,1) cnn_lstm_fit(:,1) cnn_lstm_amps(:,1) ssa(:,1) fit(:,1) baseline(:,1)];
arrs_mse = [lstm(:,2) cnn_lstm_fit(:,2) cnn_lstm_amps(:,2) ssa(:,2) fit(:,2) baseline(:,2)];
arrs_r2 = [lstm(:,3) cnn_lstm_fit(:,3) cnn_lstm_amps(:,3) ssa(:,3) fit(:,3) baseline(:,3)];

num_arrs = length(arrs_mae(1,:));
plotErrorsMatrix(arrs_mae, arrs_mse, arrs_r2, mins, windows, threshold)

plotErrorsHistogram(lstm(:,1), 'LSTM predictions', mins, windows, 'k', threshold)
plotErrorsHistogram(cnn_lstm_fit(:,1), 'CNN + LSTM (Fit) predictions', mins, windows, 'r', threshold)
plotErrorsHistogram(cnn_lstm_amps(:,1), 'CNN + LSTM (Amplitudes) predictions', mins, windows, 'b', threshold)
plotErrorsHistogram(ssa(:,1), 'SSA predictions', mins, windows, [0.5 0.2 0.8], threshold)
plotErrorsHistogram(fit(:,1), 'Fit results', mins, windows, [0.1 0.4 0.8], threshold)

close all