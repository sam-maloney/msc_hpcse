clear;

data = load('OVS_RW_dt.dat','-ascii');

dt  = data(:,1);
rms = data(:,2);

subplot(2,2,1);
h1 = loglog(dt,rms,'b.-');
xlabel('dt');
ylabel('RMS Error');
xlim([min(dt) max(dt)]);
set(gca,'XTick',[0.00001 0.0001 0.001]);
title('Time Step');

subplot(2,2,3);
p = (log(rms(2:end)) - log(rms(1:end-1)))./(log(dt(2:end)) - log(dt(1:end-1)));
h3 = semilogx(dt(2:end),p,'b.-');
xlabel('dt');
ylabel('Order of Accuracy');
xlim([min(dt) max(dt)]);
set(gca,'XTick',[0.00001 0.0001 0.001]);
ylim([0 4]);
line([min(dt) max(dt)],[2 2],'Color','black','LineStyle',':');


data = load('OVS_RW_M.dat','-ascii');

M  = data(:,1);
rms = data(:,2);

subplot(2,2,2);
h2 = loglog(M,rms,'b.-');
xlabel('M');
ylabel('RMS Error');
xlim([min(M) max(M)]);
set(gca,'XTick',[10000 100000 1000000 10000000]);
title('Number of Particles');

subplot(2,2,4);
p = (log(rms(1:end-1)) - log(rms(2:end)))./(log(M(2:end)) - log(M(1:end-1)));
h4 = semilogx(M(2:end),p,'b.-');
xlabel('M');
ylabel('Order of Accuracy');
xlim([min(M) max(M)]);
set(gca,'XTick',[10000 100000 1000000 10000000]);
ylim([0 4]);
% line([min(M) max(M)],[2 2],'Color','black','LineStyle',':');


% data = load('OVS_ADI_dh.dat','-ascii');
% 
% dh = 1./(data(:,1) - 1);
% rms = data(:,2);
% 
% subplot(2,2,2);
% h2 = loglog(dh,rms,'b.-');
% xlabel('dh');
% ylabel('RMS Error');
% xlim([min(dh) max(dh)]);
% title('Grid Spacing');
% set(gca,'XTick',[0.01 0.1]);
% 
% subplot(2,2,4);
% p = (log(rms(2:end)) - log(rms(1:end-1)))./(log(dh(2:end)) - log(dh(1:end-1)));
% h4 = semilogx(dh(2:end),p,'b.-');
% xlabel('dh');
% ylabel('Order of Accuracy');
% xlim([min(dh) max(dh)]);
% ylim([0 4]);
% line([min(dh) max(dh)],[2 2],'Color','black','LineStyle',':');
% set(gca,'XTick',[0.01 0.1]);
