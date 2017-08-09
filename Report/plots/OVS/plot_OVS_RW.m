clear;

data = load('OVS_RW_M_scalar.dat','-ascii');

M  = data(:,1);
rms = data(:,2);

subplot(2,2,1);
h2 = loglog(M,rms,'b.-');
xlabel('M');
ylabel('RMS Error');
xlim([min(M) max(M)]);
set(gca,'XTick',[1000 100000 10000000 1000000000 100000000000]);
title('Scalar');

subplot(2,2,3);
p = (log(rms(1:end-1)) - log(rms(2:end)))./(log(M(2:end)) - log(M(1:end-1)));
h4 = semilogx(M(2:end),p,'b.-');
xlabel('# of Particles, M');
ylabel('Order of Accuracy');
xlim([min(M) max(M)]);
set(gca,'XTick',[1000 100000 10000000 1000000000 100000000000]);
ylim([0 1]);
line([min(M) max(M)],[0.5 0.5],'Color','black','LineStyle',':');


data = load('OVS_RW_M_AVX.dat','-ascii');

M  = data(:,1);
rms = data(:,2);

subplot(2,2,2);
h2 = loglog(M,rms,'b.-');
xlabel('M');
ylabel('RMS Error');
xlim([min(M) 100000000000]);
ylim([1e-5 1e1]);
set(gca,'XTick',[1000 100000 10000000 1000000000 100000000000]);
set(gca,'YTick',[1e-5 1e-3 1e-1 1e1]);
title('AVX');

subplot(2,2,4);
p = (log(rms(1:end-1)) - log(rms(2:end)))./(log(M(2:end)) - log(M(1:end-1)));
h4 = semilogx(M(2:end),p,'b.-');
xlabel('# of Particles, M');
ylabel('Order of Accuracy');
xlim([min(M) 100000000000]);
set(gca,'XTick',[1000 100000 10000000 1000000000 100000000000]);
ylim([0 1]);
line([min(M) max(M)],[0.5 0.5],'Color','black','LineStyle',':');
