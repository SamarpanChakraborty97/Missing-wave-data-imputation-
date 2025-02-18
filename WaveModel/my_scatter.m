function my_scatter(truth,prediction,title_str)

figure
plot(truth,prediction,'x')
max_ax=max([abs(get(gca,'YLim')) abs(get(gca,'XLim'))]);
axis(max_ax.*[-1 1 -1 1])
hold on
plot(max_ax.*[-1 1],max_ax.*[ -1 1],'--k')
r=corr(truth,prediction);
text(-0.8.*max_ax,0.8.*max_ax,['r=' num2str(r)])
ylabel('Predicted wave height/ trough depths (m)')
xlabel('Observed wave height/ trough depths (m)')
title(title_str)
end