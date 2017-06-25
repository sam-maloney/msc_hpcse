% Plots output data from Diffusion simulations

clear;

% file_name = 'ADI_000.dat';
% file_name = 'ADI_serial.dat';
% file_name = 'ADI_scalar.dat';
% file_name = 'ADI_AVX.dat';
% file_name = 'ADI_ref.dat';
file_name = 'ADI_orig8.dat';


% file_name = 'RW_000.dat';
% file_name = 'RW_serial.dat';
% file_name = 'RW_scalar.dat';
% file_name = 'RW_ref.dat';

data = load(['Solutions/' file_name],'-ascii');
N = sqrt(length(data));

x = reshape(data(:,1),N,N)';
y = reshape(data(:,2),N,N)';
z = reshape(data(:,3),N,N)';
surf(x,y,z);

xlabel('x');
ylabel('y');