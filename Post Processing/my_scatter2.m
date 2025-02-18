function my_scatter2(truth,prediction,title_str, mins, windows, threshold)

figure;
plot(truth,prediction,'x','marker','*','markerSize',7)
max_ax=max([abs(get(gca,'YLim')) abs(get(gca,'XLim'))]);
ax = gca;
ax.FontSize = 15;
axis(max_ax.*[-1 1 -1 1])
hold on
plot(max_ax.*[-1 1],max_ax.*[ -1 1],'--k','Linewidth',2)
r=corr(truth,prediction);
text(-0.8.*max_ax,0.8.*max_ax,['r=' num2str(r)],'FontWeight','normal','FontSize',15,'interpreter','latex')
ylabel('Predicted wave height/ trough depths (m)','FontSize',15,'interpreter','latex')
xlabel('Fit wave height/ trough depths (m)','FontSize',15,'interpreter','latex')

name = sprintf(title_str + " correlation for %i mins",mins);
title(name,'interpreter','latex','FontWeight','normal','FontSize',13)

g = gcf;
g.PaperUnits = 'inches';
g.PaperPosition = [0 0 8 8];
name2 = sprintf(title_str + " correlation for %i mins over %i windows.tiff",mins,windows);
print(g,name2,'-dtiff','-r600'); 
end