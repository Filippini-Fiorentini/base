function matlab_function(file_1)
matrix = csvread(file_1);
N = sqrt(max(size(matrix)));
x_mesh = (matrix(1:N,1));
y_mesh = zeros(N,1);
z_mesh = zeros(N,N);

for i=1:N:N^2
    y_mesh(1+round(i/N)) = matrix(i,2);
end

for i=1:N
    for j=1:N
        z_mesh(i,j) = matrix(i+(N-1)*j,3);;
    end
end
figure
surf(x_mesh,y_mesh,z_mesh)