function plotBarCats(arrs, arrs_time, num_arrs, mins, windows, threshold)

Arr_M = [];
Arr_Std = [];
for i=1:num_arrs
    arr = arrs(:,i);
    mean_arr = mean(arr,'omitnan');
    std_arr = std(arr,'omitnan');
    Arr_M(i) = mean_arr;
    Arr_Std(i) = std_arr;
end

% 1:length(Arr_M)
% %length(Arr_Std)
% Arr_M+0.01
% num2str(round(Arr_M,2))
% num2str(round(Arr_Std,2))
% text(1:length(Arr_M),Arr_M+0.01,[num2str(round(Arr_M,2)) ['\pm'; '\pm'; '\pm'] num2str(round(Arr_Std,2)) [' m';' m';' m']],'vert','bottom','horiz','center');

Arr_T = [];
for i=1:num_arrs
    arr = arrs_time(:,i);
    mean_arr = mean(arr,'omitnan');
    Arr_T(i) = mean_arr;
end


X = categorical({'LSTM','CNN+LSTM(Fit)','CNN+LSTM(amplitudes)','SSA','Fit','Baseline'});
X = reordercats(X, {'LSTM','CNN+LSTM(Fit)','CNN+LSTM(amplitudes)','SSA','Fit','Baseline'});

%X = categorical({'LSTM','SSA','Fit','Baseline'});
figure
g = gcf;
bar(X,diag(Arr_M),'stacked');
hold on;
text(1:length(Arr_M),Arr_M+0.005,[num2str(round(Arr_M,2)') ['\pm'; '\pm'; '\pm'; '\pm'; '\pm'; '\pm'] num2str(round(Arr_Std,2)') [' m';' m';' m'; ' m'; ' m'; ' m']],'vert','bottom','horiz','center','FontSize',13,'FontWeight','Normal');
ax = gca;
ax.FontSize = 15;
ax.XAxis.FontName = 'Times';
ax.XAxis.FontSize = 15;
ax.YAxis.FontSize = 15;
% ax.XAxis.TickLabelRotation = 60;
ylim([0 0.5]);
ylabel('average error in m','interpreter','latex','FontSize',15)
name = sprintf('Errors for %i minute for different models over %i windows',mins,windows);
sgtitle(name,'interpreter','latex','fontweight','normal','fontsize',15);

g.PaperUnits = 'inches';
g.PaperPosition = [0 0 8 8];
name2 = sprintf("Errors for %i minute for different models over %i windows.tiff",mins,windows);
print(g,name2,'-dtiff','-r600');

figure;
f = gcf;
bar(X,diag(Arr_T),'stacked');
hold on;
text(1:length(Arr_T),Arr_T+0.005,[num2str(round(Arr_T,2)') [' sec';' sec';' sec'; ' sec'; ' sec'; ' sec']],'vert','bottom','horiz','center','FontSize',13,'FontWeight','Normal');
ax = gca;
ax.FontSize = 15;
ax.XAxis.FontName = 'Times';
ax.XAxis.FontSize = 15;
ax.YAxis.FontSize = 15;
%ax.XAxis.TickLabelRotation = 60;
ylim([0 10]);
ylabel('average error in sec','interpreter','latex','FontSize',15)
name = sprintf('Errors for %i minute for different models over %i windows',mins,windows);
sgtitle(name,'interpreter','latex','fontweight','normal','fontsize',15);

f.PaperUnits = 'inches';
f.PaperPosition = [0 0 8 8];
name2 = sprintf("Time Errors for %i minute for different models over %i windows.tiff",mins,windows);
print(f,name2,'-dtiff','-r600');
    