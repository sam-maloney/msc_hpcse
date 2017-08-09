clear;
top = 10^14;
bottom = 2*10^5;
right = 10^4;
left = 5*10^0;

N = [8 14 26 50 122 242 482 962 1922 3842];

cycles_AVX    = [984673 4466187 17933604 70529758 424078593 ...
                 1674315443 6897876151 30231108504 139269726014 728742030923];
cycles_scalar = [1928770 7297062 29534200 117667172 742727438 ...
                 3119875776 14822367366 93828448269 804670339872 3523207716584];
cycles_serial = [57510026 180314810 626389754 2323041132 13844061532 ...
                 54539402828 216986792722 874046825950 3646932527992 17157730882416];

h1 = loglog(N,cycles_AVX,'-g.');
             
hold on;

h2 = loglog(N,cycles_scalar,'-b.');
h3 = loglog(N(1:length(cycles_serial)),cycles_serial,'-r.');

xlim([left right]);
ylim([bottom top]);

% line([1 12],[1 1],'Color',[0.0 0.0 0.0]);
% line([12 12],[1 0],'Color',[0.0 0.0 0.0]);

hold off;

legh = legend([h3 h2 h1],'Baseline','Scalar','AVX','Location','East');
% vertOffset = (top - bottom)*0.02;
% set(legh,'Position',get(legh,'Position') + [0 vertOffset 0 0]);

xlabel('Grid Size N');
ylabel('Runtime [cycles]');
set(get(gca,'YLabel'),'Rotation',0);
ylabh = get(gca,'YLabel');
vertOffset = (top - bottom)*1.5;
horzOffset = (right - left)*0.00057;
set(ylabh,'Position',get(ylabh,'Position') + [horzOffset vertOffset 0]);

set(gca,'Position',get(gca,'Position') - [0 0 0 0.1]);
h = title('RW on Single Core','FontSize', 15);
vertOffset = (top - bottom)*3.5;
horzOffset = (right - left)*-0.02075;
set(h,'Position',get(h,'Position') + [horzOffset vertOffset 0]);

set(gcf,'color','w');