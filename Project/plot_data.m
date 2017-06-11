clear;

file_name = 'density_serial.dat';

data = load(file_name,'-ascii');
N = sqrt(length(data));

x = reshape(data(:,1),N,N)';
y = reshape(data(:,2),N,N)';
z = reshape(data(:,3),N,N)';
surf(x,y,z);

xlabel('x');
ylabel('y');