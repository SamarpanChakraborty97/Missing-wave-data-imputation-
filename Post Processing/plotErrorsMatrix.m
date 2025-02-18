function plotErrorsMatrix(arr1, arr2, arr3, mins, windows, threshold)

num_arrs = length(arr1(1,:));
for i=1:num_arrs
    arr = arr1(:,i);
    mean_arr = mean(arr,'omitnan');
    Arr_M(1,i) = mean_arr;
end

for i=1:num_arrs
    arr = arr2(:,i);
    mean_arr = mean(arr,'omitnan');
    Arr_M(2,i) = mean_arr;
end

for i=1:num_arrs
    arr = arr3(:,i);
    mean_arr = mean(arr,'omitnan');
    Arr_M(3,i) = -mean_arr;
end

figure;
g = gcf;

xvalues = {'LSTM','CNN + LSTM(Fit)','CNN + LSTM(amplitudes)','SSA','Fit','Baseline'};
yvalues = {'MAE','MSE','-R2 score'};
h = heatmap(xvalues,yvalues,Arr_M);
h.ColorScaling = 'scaledrows';
h.FontName = 'Times';
h.FontSize = 15;
h.Colormap = hot;
% ax = gca;
% ax.XAxis.FontName = 'Times';
% ax.XAxis.FontSize = 10;
h.XLabel = 'Different models';
h.YLabel = 'Errors';
name = sprintf('Errors for %i minute for different models over %i windows',mins,windows);
sgtitle(name,'interpreter','latex','fontweight','normal','fontsize',15);

g.PaperUnits = 'inches';
g.PaperPosition = [0 0 8 8];
name2 = sprintf("Errors for %i minute for different models over %i windows_Matrix.tiff",mins,windows);
print(g,name2,'-dtiff','-r600'); 
