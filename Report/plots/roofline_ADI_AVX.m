clear;
left = -7;
right = 8;
bottom = -2;
top = 5;

memory_bandwidth = 68; % GB/s
clock_speed = 2.6; % GHz
scalar_peak = 4;
vector_peak = 4*scalar_peak;

% beta for main memory
betaRAM = memory_bandwidth*2^30/(clock_speed*10^9);

% betas for caches
betaL3 = 64/6.3 + 64/8.4;
betaL2 = 64/6.1 + 64/2.2;
betaL1 = 64/0.5;

I = [2^left 2^right];
MRAM = betaRAM*I;
ML3 = betaL3*I;
ML2 = betaL2*I;
ML1 = betaL1*I;
pi_s = [scalar_peak scalar_peak];
pi_v = [vector_peak vector_peak];

N = [10 18 34 66 122 242 482 962 1922 3842];
perf = [1.925 2.064 2.310 2.045 1.958...
        1.795 1.746 1.252 0.928 0.819];
I_L2  = [2.023 7.287 11.219 0.050 0.028...
         0.019 0.019 0.017 0.016 0.015];
I_L3  = [4.386 13.860 49.258 146.745 0.411...
         0.068 0.063 0.036 0.027 0.025];
I_RAM = [17.762 62.560 256.690 1007.828 3582.589...
         13416.279 26625.866 216.532 0.119 0.097];

hold on;

xlim([left right]);
ylim([bottom top]);

xticks = (left+1):3:(right-1);
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
plot(log2(I),log2(ML2), '-k');
% plot(log2(I),log2(ML1), '-k');

plot(log2(I_L3),log2(perf),'-b.');

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
horzOffset = (right - left)*0.305;
set(ylabh,'Position',get(ylabh,'Position') + [horzOffset vertOffset 0]);

set(gca,'Position',get(gca,'Position') - [0 0 0 0.1]);
h = title('Xeon E5-2690 v3','FontSize', 15);
vertOffset = (top - bottom)*0.08;
horzOffset = (right - left)*-0.335;
set(h,'Position',get(h,'Position') + [horzOffset vertOffset 0]);

set(gcf,'color','w');