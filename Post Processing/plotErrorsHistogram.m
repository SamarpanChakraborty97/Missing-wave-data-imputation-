function plotErrorsHistogram(arr, title_str, mins, windows, col, threshold)
%colors = ['b','r','k',[0.5 0.2 0.8],[0.1 0.4 0.8],[0.9 0.4 0.1]];

figure;
g = gcf;
idx = find(abs(arr)>threshold);
arr(idx) = [];
h = histogram(arr);
h.BinWidth = 0.05;
h.Normalization = 'probability';
h.FaceColor = col;
h.EdgeColor = 'k';
ax = gca;
ax.FontSize = 15;
ax.XAxis.FontName = 'Times';
ax.XAxis.FontSize = 15;
ax.YAxis.FontSize = 15;
xlabel('errors in metres','interpreter','latex','FontSize',15);
ylabel('probability of errors','interpreter','latex','FontSize',15);
name3 = sprintf(title_str + " error histogram for %i mins",mins);
title(name3,'interpreter','latex','FontWeight','normal','FontSize',13)

g.PaperUnits = 'inches';
g.PaperPosition = [0 0 8 8];
name4 = sprintf(title_str + " error histogram for %i mins over %i windows till %1.2f m.tiff",mins,windows,threshold);
print(g,name4,'-dtiff','-r600');
end
    
    