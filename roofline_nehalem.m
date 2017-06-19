clear;
left = -5;
right = 7;
bottom = -1;
top = 3;

memory_bandwidth = 21; % GB/s
clock_speed = 1.6; % GHz
scalar_peak = 2;
vector_peak = 2*scalar_peak;

% beta for main memory
betaRAM = memory_bandwidth*2^30/(clock_speed*10^9);

% beta for L3 cache
betaL3 = 64/5.75 + 64/10.4;

I = [2^left 2^right];
MRAM = betaRAM*I;
ML3 = betaL3*I;
pi_s = [scalar_peak scalar_peak];
pi_v = [vector_peak vector_peak];

hold on;

xlim([left right]);
ylim([bottom top]);

xticks = (left+1):2:(right-1);
set(gca, 'XTick', xticks);
for j = 1:length(xticks)
  xtl{j} = num2str(2^xticks(j));
end
set(gca, 'XTickLabel', xtl);

yticks = bottom:1:top;
set(gca, 'YTick', yticks);
for j = 1:length(yticks)
  ytl{j} = num2str(2^yticks(j));
end
set(gca, 'YTickLabel', ytl);

% Obtain the tick mark locations
xtick = get(gca,'XTick'); 
xtick = [left xtick right];
% Obtain the limits of the y axis
ylims = get(gca,'Ylim');
% Create line data
X = repmat(xtick,2,1);
Y = repmat(ylims',1,size(xtick,2));
% Plot line data
h = plot(X,Y,'-k');
set(h,'Color',[0.9 0.9 0.9]);

% Obtain the tick mark locations
ytick = get(gca,'YTick');
% Obtain the limits of the x axis
xlims = get(gca,'Xlim');
% Create line data
Y = repmat(ytick,2,1);
X = repmat(xlims',1,size(ytick,2));
% Plot line data
h = plot(X,Y,'-k');
set(h,'Color',[0.9 0.9 0.9]);

% line([left left],[bottom top],'Color',[0.8 0.8 0.8]);
% line([left right],[bottom bottom],'Color',[0.8 0.8 0.8]);
% line([left right],[top top],'Color',[0.8 0.8 0.8]);
% line([right right],[top bottom],'Color',[0.8 0.8 0.8]);

% plot rooflines
plot(log2(I),log2(pi_s),'-k');
plot(log2(I),log2(pi_v),'-k');
plot(log2(I),log2(MRAM),'-k');
plot(log2(I),log2(ML3), '-k');


% h = text(log2(0.125),log2(4.4),'L3 Bandwidth');
% set(h, 'rotation', 57);
% 
% h = text(log2(0.52),log2(4.1),'RAM Bandwidth');
% set(h, 'rotation', 58);
% 
% text(log2(270),log2(2.5),'Scalar Peak');
% text(log2(390),log2(4.5),'AVX Peak');

hold off;

% legh = legend([h3 h2 h1],'AVX + ILP','AVX',...
%                   'Scalar','Location','SouthEast');
% vertOffset = (top - bottom)*0.02;
% set(legh,'Position',get(legh,'Position') + [0 vertOffset 0 0]);

xlabel('Operational Intensity [flops/byte]');
ylabel('Performance [flops/cycle]');
set(get(gca,'YLabel'),'Rotation',0);
ylabh = get(gca,'YLabel');
vertOffset = (top - bottom)*0.53;
horzOffset = (right - left)*0.3;
set(ylabh,'Position',get(ylabh,'Position') + [horzOffset vertOffset 0]);

set(gca,'Position',get(gca,'Position') - [0 0 0 0.1]);
h = title('Core i7-720QM','FontSize', 15);
vertOffset = (top - bottom)*0.08;
horzOffset = (right - left)*-0.305;
set(h,'Position',get(h,'Position') + [horzOffset vertOffset 0]);

set(gcf,'color','w');