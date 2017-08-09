clear;

data = load('OVS_ADI_dt.dat','-ascii');

dt  = data(:,1);
rms = data(:,2);

subplot(2,2,1);
h1 = loglog(dt,rms,'b.-');
xlabel('dt');
ylabel('RMS Error');
xlim([0.000001 0.001]);
ylim([1e-6 1])
set(gca,'XTick',[0.000001 0.00001 0.0001 0.001]);
set(gca,'YTick',[0.000001 0.0001 0.01 1]);
title('Time Step');

subplot(2,2,3);
p = (log(rms(2:end)) - log(rms(1:end-1)))./(log(dt(2:end)) - log(dt(1:end-1)));
h3 = semilogx(dt(2:end),p,'b.-');
xlabel('dt');
ylabel('Order of Accuracy');
xlim([0.000001 0.001]);
set(gca,'XTick',[0.000001 0.00001 0.0001 0.001]);
ylim([0 4]);
line([min(dt) max(dt)],[2 2],'Color','black','LineStyle',':');

data = load('OVS_ADI_dh_2.dat','-ascii');

dh = 1./(data(:,1) - 1);
rms = data(:,2);

subplot(2,2,2);
h2 = loglog(dh,rms,'b.-');
xlabel('dh');
ylabel('RMS Error');
xlim([min(dh) max(dh)]);
ylim([1e-11 1e-5]);
set(gca,'YTick',[1e-11 1e-9 1e-7 1e-5]);
title('Grid Spacing');
set(gca,'XTick',[0.01 0.1]);

subplot(2,2,4);
p = (log(rms(2:end)) - log(rms(1:end-1)))./(log(dh(2:end)) - log(dh(1:end-1)));
h4 = semilogx(dh(2:end),p,'b.-');
xlabel('dh');
ylabel('Order of Accuracy');
xlim([min(dh) max(dh)]);
ylim([0 4]);
line([min(dh) max(dh)],[2 2],'Color','black','LineStyle',':');
set(gca,'XTick',[0.01 0.1]);
