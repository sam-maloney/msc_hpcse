clear;
top = 10^13;
bottom = 2*10^5;
right = 10^4;
left = 5*10^0;

N = [8 14 26 50 122 242 482 962 1922 3842];

cycles_AVX    = [984673 4466187 17933604 70529758 424078593 ...
                 1674315443 6897876151 30231108504 139269726014 728742030923];
cycles_scalar = [1928770 7297062 29534200 117667172 742727438 ...
                 3119875776 14822367366 93828448269 804670339872 3523207716584];
cycles_serial = [2819604252 30515687408 392094146074 5581455849694];

loglog(N,cycles_AVX,'-g.');
             
hold on;

loglog(N,cycles_scalar,'-b.');
loglog(N(1:length(cycles_serial)),cycles_serial,'-r.');

xlim([left right]);
ylim([bottom top]);

% line([1 12],[1 1],'Color',[0.0 0.0 0.0]);
% line([12 12],[1 0],'Color',[0.0 0.0 0.0]);

hold off;

% legh = legend([h3 h2 h1],'AVX + ILP','AVX',...
%                   'Scalar','Location','SouthEast');
% vertOffset = (top - bottom)*0.02;
% set(legh,'Position',get(legh,'Position') + [0 vertOffset 0 0]);

xlabel('Grid Size N');
ylabel('Runtime [cycles]');
set(get(gca,'YLabel'),'Rotation',0);
ylabh = get(gca,'YLabel');
vertOffset = (top - bottom)*1.1;
horzOffset = (right - left)*0.0006;
set(ylabh,'Position',get(ylabh,'Position') + [horzOffset vertOffset 0]);

set(gca,'Position',get(gca,'Position') - [0 0 0 0.1]);
h = title('RW on Single Core','FontSize', 15);
vertOffset = (top - bottom)*3.0;
horzOffset = (right - left)*-0.0207;
set(h,'Position',get(h,'Position') + [horzOffset vertOffset 0]);

set(gcf,'color','w');