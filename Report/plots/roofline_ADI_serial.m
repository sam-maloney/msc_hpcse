clear;
left = -6;
right = 9;
bottom = -2;
top = 5;

% memory_bandwidth = 68; % GB/s
memory_bandwidth = (11000+14500)/2^10; % GB/s
clock_speed = 2.6; % GHz
scalar_peak = 4;
vector_peak = 4*scalar_peak;

% beta for main memory
betaRAM = memory_bandwidth*2^30/(clock_speed*10^9);

% betas for caches
betaL3 = 64/6.3 + 64/8.4;
betaL2 = 64/6.1 + 64/2.2;
betaL1 = 12*8;

I = [2^left 2^right];
MRAM = betaRAM*I;
ML3  = betaL3*I;
ML2  = betaL2*I;
ML1  = betaL1*I;
pi_s = [scalar_peak scalar_peak];
pi_v = [vector_peak vector_peak];

N = [10 18 34 66 122 242 482 962 1922 3842 7682];

perf_AVX  = [1.925 2.064 2.310 2.045 1.958...
             1.795 1.746 1.252 0.928 0.819 0.801];
I_L2_AVX  = [20.229 72.867 112.189 0.499 0.280...
             0.193 0.185 0.174 0.165 0.151 0.140];
I_L3_AVX  = [43.860 138.595 492.583 1467.446 4.109...
             0.681 0.631 0.363 0.274 0.246 0.220];
I_RAM_AVX = [177.625 625.597 2566.898 10078.278 35825.893...
             134162.791 266258.661 2165.317 1.192 0.974 0.963];
         
perf_scalar  = [1.549 1.720 1.693 1.670 1.625...
                1.417 1.315 0.968 0.669 0.640 0.638];
I_L2_scalar  = [19.604 63.431 26.669 0.431 0.244...
                0.159 0.151 0.143 0.137 0.125 0.111];
I_L3_scalar  = [39.736 141.312 483.390 1178.315 9.138...
                0.579 0.389 0.275 0.221 0.200 0.169];
I_RAM_scalar = [159.978 571.003 2232.055 8825.832 29795.005...
                119985.741 414843.750 2492.483 1.036 0.837 0.842];
         
perf_serial  = [0.734 0.526 0.480 0.459 0.449...
                0.443 0.441 0.374 0.333 0.305 0.274];
I_L2_serial  = [38.730 139.276 283.299 1.017 0.797...
                0.113 0.139 0.111 0.095 0.075 0.084];
I_L3_serial  = [77.529 319.366 1040.724 3515.800 173.098...
                1.444 0.379 0.629 0.132 0.111 0.118];
I_RAM_serial = [324.468 1279.427 5348.837 20733.198 73016.277...
                188433.814 989767.842 16011.485 2.335 1.913 1.941];

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
plot(log2(I),log2(MRAM),'-r');
plot(log2(I),log2(ML3), '-b');
plot(log2(I),log2(ML2), '-g');
% plot(log2(I),log2(ML1), '-k');

fac = 1/4;

% h1 = plot(log2(fac*I_RAM_AVX),log2(perf_AVX),'-r.');
% h1 = plot(log2(fac*I_RAM_scalar),log2(perf_scalar),'-r.');
h1 = plot(log2(fac*I_RAM_serial),log2(perf_serial),'-r.');

% h2 = plot(log2(fac*I_L3_AVX),log2(perf_AVX),'-b.');
% h2 = plot(log2(fac*I_L3_scalar),log2(perf_scalar),'-b.');
h2 = plot(log2(fac*I_L3_serial),log2(perf_serial),'-b.');

% h3 = plot(log2(fac*I_L2_AVX),log2(perf_AVX),'-g.');
% h3 = plot(log2(fac*I_L2_scalar),log2(perf_scalar),'-g.');
h3 = plot(log2(fac*I_L2_serial),log2(perf_serial),'-g.');

% h = text(log2(0.125),log2(4.4),'L3 Bandwidth');
% set(h, 'rotation', 57);
% 
% h = text(log2(0.52),log2(4.1),'RAM Bandwidth');
% set(h, 'rotation', 58);

text(log2(4),log2(4.7),'Scalar Peak');
text(log2(5),log2(18.5),'AVX Peak');

hold off;

legh = legend([h3 h2 h1],'L2 cache','L3 cache','RAM','Location','NorthEast');
vertOffset = (top - bottom)*-0.03;
set(legh,'Position',get(legh,'Position') + [0 vertOffset 0 0]);

xlabel('Operational Intensity [flops/byte]');
ylabel('Performance [flops/cycle]');
set(get(gca,'YLabel'),'Rotation',0);
ylabh = get(gca,'YLabel');
vertOffset = (top - bottom)*0.52;
horzOffset = (right - left)*0.305;
set(ylabh,'Position',get(ylabh,'Position') + [horzOffset vertOffset 0]);

set(gca,'Position',get(gca,'Position') - [0 0 0 0.1]);
h = title('ADI Roofline, Baseline','FontSize', 15);
vertOffset = (top - bottom)*0.08;
horzOffset = (right - left)*-0.296;
set(h,'Position',get(h,'Position') + [horzOffset vertOffset 0]);

set(gcf,'color','w');