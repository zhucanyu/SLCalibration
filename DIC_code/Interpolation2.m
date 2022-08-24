function deformed_subset_sub_pixel=Interpolation2(image_deformed,P_vector,x_pos,y_pos,r)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
deformed_subset = zeros(2*r+1,2*r+1);
% deformed_subset
U = P_vector(1,1);
U_x = P_vector(2,1);
U_y = P_vector(3,1);
V = P_vector(4,1);
V_x = P_vector(5,1);
V_y = P_vector(6,1);
center_x = r+ 1;
center_y = r+ 1;
center_x = double(center_x); 
center_y = double(center_y);
for i=1:2*r+1  % y
    for j=1:2*r+1  % x
   
        current_delta_x = double(j)-center_x;
        current_delta_y = double(i)-center_y;
        % 下面求出的是变形子区的坐标，还需插值出该点的灰度值和一阶梯度值
        x_pos = double(x_pos);
        y_pos = double(y_pos);
        deformed_x = double(x_pos + U + U_x * current_delta_x + U_y * current_delta_y + current_delta_x) ;
        deformed_y = double(y_pos + V + V_x * current_delta_x + V_y * current_delta_y + current_delta_y) ;
        X_grid = zeros(6,6);
        Y_grid = zeros(6,6);
        Gray_value_grid = zeros(6,6);
        int_deformed_x = floor(deformed_x);
        int_deformed_y = floor(deformed_y);
        
        % 获取该点的零阶和一阶梯度,调用函数interp2_mehod
        for k = int_deformed_y-2:int_deformed_y+3
            for l = int_deformed_x-2:int_deformed_x+3
                index_k = k-int_deformed_y + 3;
                index_l = l-int_deformed_x + 3;
                X_grid(index_k,index_l) = l;
                Y_grid(index_k,index_l) = k;
                Gray_value_grid(index_k,index_l) = image_deformed(k,l);                
            end
        end
    
        % interp2_mehod 函数调用
    
        [v1,v2,v3] =interpolation(X_grid,Y_grid,Gray_value_grid,deformed_x,deformed_y);
        deformed_subset(i,j) = v1;
          
    end
end

deformed_subset_sub_pixel = deformed_subset;
end

