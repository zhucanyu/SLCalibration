function [ Gray_result,diff_x_result , diff_y_result] = interpolation(X_grid , Y_grid , Z_value , interp_x_pos , interp_y_pos)
% 插值，X Y 为网格布局，Z为灰度值，X_grid Y_grid 为4 x 4 网格
% 原函数的值为 interp2 求解，而一阶偏导数则需要曲面拟合方法求解

[m,n] = size(Z_value);
total_point = m*n;

best_fit_A = zeros(total_point,10);
best_fit_B = zeros(total_point,1);
syms x y S;

S = [x*x*x,x*x*y,x*y*y,y*y*y, x*x , x*y , y*y , x , y ,1];

for i=1:m
    for j=1:n
        
        x_pos = double(X_grid(i,j));
        y_pos = double(Y_grid(i,j));
        z_value = double(Z_value(i,j));
        S_cache = subs(S,{x,y},{x_pos,y_pos});
        index_1D = (i-1)*n + j;
        best_fit_A(index_1D,:) = S_cache;
        best_fit_B(index_1D,:) = z_value;      
    end
end

best_fit_X = best_fit_A \ best_fit_B;

a = best_fit_X(1,1);
b = best_fit_X(2,1);
c = best_fit_X(3,1);
d = best_fit_X(4,1);
e = best_fit_X(5,1);
f = best_fit_X(6,1);
g = best_fit_X(7,1);
h = best_fit_X(8,1);
I = best_fit_X(9,1);
J = best_fit_X(10,1);

x = double(interp_x_pos);
y = double(interp_y_pos);
    Gray_result = a*x*x*x + b*x*x*y + c*x*y*y + d*y*y*y + e*x*x + f*x*y + g*y*y + h*x + I*y + J;
    diff_x_result = 3*a*x*x + 2*b*x*y + c*y*y + 2*e*x + f*y + h ;
    diff_y_result = b*x*x + 2*c*x*y + 3*d*y*y + f*x + 2*g*y + I ;



end
