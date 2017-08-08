clear;

data = load('OVS_RW_M_serial.dat','-ascii');

M  = data(:,1);
rms = data(:,2);

subplot(1,2,1);
h2 = loglog(M,rms,'b.-');
xlabel('# of Particles, M');
ylabel('RMS Error');
xlim([min(M) 1000000000]);
set(gca,'XTick',[1000 100000 10000000 1000000000]);
title('Baseline');

subplot(1,2,2);
p = (log(rms(1:end-1)) - log(rms(2:end)))./(log(M(2:end)) - log(M(1:end-1)));
h4 = semilogx(M(2:end),p,'b.-');
xlabel('# of Particles, M');
ylabel('Order of Accuracy');
xlim([min(M) 1000000000]);
set(gca,'XTick',[1000 100000 10000000 1000000000]);
ylim([0 1]);
line([min(M) max(M)],[0.5 0.5],'Color','black','LineStyle',':');