clc
clear all

mins = 1;

%%%%%% Minimum number of peaks %%%%%%
lstm_min = readmatrix('lstm_min_time_series.txt');
cnn_lstm_min = readmatrix('cnn_lstm_min_time_series.txt');
ssa_min = readmatrix('ssa_min_time_series.txt');
fit_min = readmatrix('fit_min_time_series.txt');
truth_min = readmatrix('true_min_time_series.txt');

L = length(truth_min);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% Maximum number of peaks %%%%%%
lstm_max = readmatrix('lstm_max_time_series.txt');
cnn_lstm_max = readmatrix('cnn_lstm_max_time_series.txt');
ssa_max = readmatrix('ssa_max_time_series.txt');
fit_max = readmatrix('fit_max_time_series.txt');
truth_max = readmatrix('true_max_time_series.txt');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%