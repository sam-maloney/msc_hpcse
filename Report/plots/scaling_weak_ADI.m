clear;
top = 1;
bottom = 0;
right = 12;
left = 1;

cores = [1 2 4 6 8 10 12];

efficiency = [1.000 0.931 0.810 0.803 0.728 0.592 0.419];

hold on;

plot(cores,efficiency,'-b.');

xlim([1 12]);
ylim([0 1]);
set(gca,'XTick',[1 2 4 6 8 10 12])

line([1 12],[1 1],'Color',[0.0 0.0 0.0]);
line([12 12],[1 0],'Color',[0.0 0.0 0.0]);

hold off;

% legh = legend([h3 h2 h1],'AVX + ILP','AVX',...
%                   'Scalar','Location','SouthEast');
% vertOffset = (top - bottom)*0.02;
% set(legh,'Position',get(legh,'Position') + [0 vertOffset 0 0]);

xlabel('Number of Cores');
ylabel('Efficiency');
set(get(gca,'YLabel'),'Rotation',0);
ylabh = get(gca,'YLabel');
vertOffset = (top - bottom)*0.53;
horzOffset = (right - left)*0.12;
set(ylabh,'Position',get(ylabh,'Position') + [horzOffset vertOffset 0]);

set(gca,'Position',get(gca,'Position') - [0 0 0 0.1]);
h = title('ADI Weak Scaling, N = 7680','FontSize', 15);
vertOffset = (top - bottom)*0.08;
horzOffset = (right - left)*-0.178;
set(h,'Position',get(h,'Position') + [horzOffset vertOffset 0]);

set(gcf,'color','w');