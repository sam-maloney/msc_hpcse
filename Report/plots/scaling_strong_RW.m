clear;
top = 24;
bottom = 1;
right = 24;
left = 1;

cores = [1 2 4 6 8 10 12 16 20 24];

speedup = [1.000 1.996 3.516 5.315 6.984 8.626 10.214 10.065 11.894 13.234];

hold on;

plot(cores,cores,'-k');
plot(cores,speedup,'-b.');

xlim([1 24]);
ylim([1 24]);
set(gca,'XTick',[1 4 8 12 16 20 24])
set(gca,'YTick',[1 4 8 12 16 20 24])

line([12.5 12.5],[1 24],'Color',[0.5 0.5 0.5],'LineStyle','--');
line([1 24],[24 24],'Color',[0.0 0.0 0.0]);
line([24 24],[24 1],'Color',[0.0 0.0 0.0]);

h = text(5.5,22,'No Hyper-threading          Hyper-threading');

hold off;

% legh = legend([h3 h2 h1],'AVX + ILP','AVX',...
%                   'Scalar','Location','SouthEast');
% vertOffset = (top - bottom)*0.02;
% set(legh,'Position',get(legh,'Position') + [0 vertOffset 0 0]);

xlabel('Number of Cores');
ylabel('Speedup Factor');
set(get(gca,'YLabel'),'Rotation',0);
ylabh = get(gca,'YLabel');
vertOffset = (top - bottom)*0.52;
horzOffset = (right - left)*0.155;
set(ylabh,'Position',get(ylabh,'Position') + [horzOffset vertOffset 0]);

set(gca,'Position',get(gca,'Position') - [0 0 0 0.1]);
h = title('RW Strong Scaling, N = 7682','FontSize', 15);
vertOffset = (top - bottom)*0.08;
horzOffset = (right - left)*-0.18;
set(h,'Position',get(h,'Position') + [horzOffset vertOffset 0]);

set(gcf,'color','w');