function plotBarMax(arrs,num_arrs, mins, windows)

Arr_M = [];
for i=1:num_arrs
    arr = arrs(:,i);
    mean_arr = mean(arr,'omitnan');
    Arr_M(i) = mean_arr;
end

X = categorical({'LSTM','CNN+LSTM-Fit','CNN+LSTM-Slow Varying Amplitudes','SSA','Fit','Baseline'});
X = reordercats(X, {'LSTM','CNN+LSTM-Fit','CNN+LSTM-Slow Varying Amplitudes','SSA','Fit','Baseline'});

%X = categorical({'LSTM','SSA','Fit','Baseline'});
figure
g = gcf;

%subplot(1,1,1)
bar(X,diag(Arr_M),'stacked');
ax = gca;
ax.FontSize = 15;
ax.XAxis.FontName = 'Times';
ax.XAxis.FontSize = 15;
ax.YAxis.FontSize = 15;
%ax.XAxis.TickLabelRotation = 60;
ylabel('average maximal height/depths error (m)','interpreter','latex','FontSize',15)

name = sprintf('Errors for the highest crests/ troughs in %i minute over %i windows',mins, windows);
sgtitle(name,'interpreter','latex','fontweight','normal','fontsize',15);

g.PaperUnits = 'inches';
g.PaperPosition = [0 0 8 8];
name2 = sprintf('Errors for the highest crests in %i minute for different models.tiff',mins);
print(g,name2,'-dtiff','-r600'); 
