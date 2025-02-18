clear all
close all
clc

src1='071p1_d19.nc';
src2='076p1_d19.nc';
src3='222p1_d02.nc';

%%%%%% Buoy 71 %%%%%%%
% ncdisp(src1)
% 
% info=ncinfo(src1,'xyzZDisplacement');
% Starttime=ncread(src1,'xyzStartTime');
% samplerate=ncread(src1,'xyzSampleRate');
% filt_delay=ncread(src1,'xyzFilterDelay');
% BeginDate=datetime(1970,1,1)+seconds(Starttime-filt_delay)
% EndDate=datetime(1970,1,1)+seconds(Starttime-filt_delay)+seconds(info.Size/samplerate)
%%%%%%%%%%%%%%%%%%%%%%

%%%%%% Buoy 76 %%%%%%%
%ncdisp(src2);

info=ncinfo(src2,'xyzZDisplacement');
Starttime=ncread(src2,'xyzStartTime');
samplerate=ncread(src2,'xyzSampleRate');
filt_delay=ncread(src2,'xyzFilterDelay');
BeginDate=datetime(1970,1,1)+seconds(Starttime-filt_delay)
EndDate=datetime(1970,1,1)+seconds(Starttime-filt_delay)+seconds(info.Size/samplerate)
%%%%%%%%%%%%%%%%%%%%%%

%%%%%% Buoy 222 %%%%%%
% ncdisp(src3)
% 
% info=ncinfo(src3,'xyzZDisplacement');
% Starttime=ncread(src3,'xyzStartTime');
% samplerate=ncread(src3,'xyzSampleRate');
% filt_delay=ncread(src3,'xyzFilterDelay');
% BeginDate=datetime(1970,1,1)+seconds(Starttime-filt_delay);
% EndDate=datetime(1970,1,1)+seconds(Starttime-filt_delay)+seconds(info.Size/samplerate)
%%%%%%%%%%%%%%%%%%%%%