clear;
top = 5*10^12;
bottom = 2*10^5;
right = 10^4;
left = 7*10^0;

N = [10 18 34 66 122 242 482 962 1922 3842];

cycles_AVX    = [556954 2030545 7175058 32227887 118033356 ...
                 514134617 2113379757 11778571009 63603267703 288138307328];
cycles_scalar = [609455 2140148 8580912 34565558 124533788 ...
                 570110102 2455958232 13341730856 77109120432 322608921400];
cycles_serial = [2660340 15212056 67465372 283717286 1023417704 ...
                 4149885946 16714511968 78836296474 354014868042 1545784109690];

loglog(N,cycles_AVX,'-g.');
             
hold on;

loglog(N,cycles_scalar,'-b.');
loglog(N,cycles_serial,'-r.');

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
horzOffset = (right - left)*0.00075;
set(ylabh,'Position',get(ylabh,'Position') + [horzOffset vertOffset 0]);

set(gca,'Position',get(gca,'Position') - [0 0 0 0.1]);
h = title('ADI on Single Core','FontSize', 15);
vertOffset = (top - bottom)*3.0;
horzOffset = (right - left)*-0.0243;
set(h,'Position',get(h,'Position') + [horzOffset vertOffset 0]);

set(gcf,'color','w');