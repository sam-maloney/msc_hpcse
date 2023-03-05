clear;

data = load('OVS_ADI_dt_new256_2.dat','-ascii');

dt  = data(:,1);
rms = data(:,2);

subplot(2,2,1);
h1 = loglog(dt,rms,'b.-');
xlabel('dt');
ylabel('RMS Error');
xlim([0.0004 0.6]);
ylim([10^-13 max(rms)])
set(gca,'XTick',[0.0001 0.001 0.01 0.1 1]);
set(gca,'YTick',[10^-13 10^-10 10^-7 10^-4]);
title('Time Step');

subplot(2,2,3);
p = (log(rms(2:end)) - log(rms(1:end-1)))./(log(dt(2:end)) - log(dt(1:end-1)));
% p = real(log((rms(3:end) - rms(2:end-1))./(rms(2:end-1) - rms(1:end-2))))/...
%     log(dt(2)/dt(1));
h3 = semilogx(dt(2:1+length(p)),p,'b.-');
xlabel('dt');
ylabel('Order of Accuracy');
xlim([0.0004 0.6]);
set(gca,'XTick',[ 0.0001 0.001 0.01 0.1 1]);
ylim([0 4]);
line([min(dt) max(dt)],[2 2],'Color','black','LineStyle',':');

data = load('OVS_ADI_dh_new_root2.dat','-ascii');

dh = 1./(data(:,1) - 1);
rms = data(:,2);

subplot(2,2,2);
h2 = loglog(dh,rms,'b.-');
xlabel('dh');
ylabel('RMS Error');
xlim([min(dh) max(dh)]);
ylim([1e-13 max(rms)])
set(gca,'YTick',[1e-13 1e-12 1e-11 1e-10]);
title('Grid Spacing');
set(gca,'XTick',[0.01 0.1]);

subplot(2,2,4);
p = (log(rms(2:end)) - log(rms(1:end-1)))./(log(dh(2:end)) - log(dh(1:end-1)));
% p = real(log((rms(3:end) - rms(2:end-1))./(rms(2:end-1) - rms(1:end-2))))/...
%     log(dh(2)/dh(1));
h4 = semilogx(dh(2:1+length(p)),p,'b.-');
xlabel('dh');
ylabel('Order of Accuracy');
xlim([min(dh) max(dh)]);
ylim([0 4]);
line([min(dh) max(dh)],[2 2],'Color','black','LineStyle',':');
set(gca,'XTick',[0.01 0.1]);
