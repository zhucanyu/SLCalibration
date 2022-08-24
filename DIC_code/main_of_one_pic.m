clc
close all
clear all

% it is a DIC code to find matching coordinates of one 'centers*.csv' in one
% '*.bmp'in 'speckle.bmp' and save them as 'centersnew*.csv' ,you can use
% it to test if the code work
% 主函数主要读取要匹配的两张图片，进行计算

files = dir(fullfile('board_pics\','*.bmp'));
filesd = dir(fullfile('speckle_pic\','*.bmp'));
lengthFiles = length(files);

r=30;%定义子区半径 (subsets radius)
search_radius = 200;%定义搜索半径 (search radius)
count = 1 ;%计算图片编号 （Number of the image being calculated）

ei  = 1;bi=1;
err=nan;bug=nan;
filename   =   strcat('centers',num2str(count),'.csv');
image_reference = uint8(imread(strcat('board_pics\',files(2*count-1).name)));%文件所在路径

figure(count);
imshow(image_reference);
title(strcat('原始图像',num2str(count)));
hold on
%读入参考图中识别出的圆点中心坐标作为计算中心点
centers=csvread(filename);

POI_point_count = size(centers,1); 
x_pos1(:)=NaN;
y_pos1(:)=NaN;

for i=1:POI_point_count      %画出计算点中心位置
            x_pos1(i) =centers(i,1);
            y_pos1(i) =centers(i,2);
           plot(x_pos1,y_pos1,'*b');       
end
hold off

 
filename   =   strcat('centers',num2str(count),'.csv');

image_reference = uint8(imread(strcat('board_pics\',files(2*count-1).name)));%文件所在路径
image_deformed = uint8(imread(strcat('speckle_pic\',filesd(1).name)));
figure(count);
hold on
[height, width]=size(image_reference);%取出图片大小

centers=csvread(filename); %读入参考图中识别出的圆点中心坐标作为计算中心点
POI_point_count = size(centers,1); 
x_pos1(POI_point_count)=NaN;
y_pos1(POI_point_count)=NaN;

for i=1:POI_point_count      %给计算点中心标记
            x_pos1(i) =centers(i,1);
            y_pos1(i) =centers(i,2);          
end
for j=1:POI_point_count
text(centers(j,1),centers(j,2),num2str(j),'color','y');
end

figure(count+100);
imshow(image_deformed);
title('基准图像');
hold on

result_matx(POI_point_count)=NaN;
result_maty(POI_point_count)=NaN;

x_pos(POI_point_count)=NaN;
y_pos(POI_point_count)=NaN;
    for i=1:POI_point_count   %使得参考值取位于整像素位置，避免在开始引入插值误差
     x_pos(i)=fix(x_pos1(i));
     y_pos(i)=fix(y_pos1(i));
    end

    
   for i=1:POI_point_count
         refer_subset = zeros(2*r+1,2*r+1);
         deformed_subset = zeros(2*r+1 ,2*r+1);
          normb=10;   
            % 参考模板的灰度矩阵
            for m = y_pos(i) -r : y_pos(i) +r
                for n = x_pos(i) -r : x_pos(i) +r
                    patch_m = m - y_pos(i) + r + 1;
                    patch_n = n - x_pos(i) + r + 1;
                    gray_level = image_reference(m,n);
                    refer_subset(patch_m , patch_n) =gray_level;
                end
            end
            m=NaN;
            n=NaN;
            % 搜索范围内部进行匹配
            correlation_coefficient_mat_per_point = zeros(2*search_radius+1,2*search_radius+1); % 存储搜索范围内的相关值矩阵，并比较其中最小的，即为整像素匹配点
            for re_deformed_subset_center_y = y_pos(i) - search_radius : y_pos(i) + search_radius % 搜索范围
                 for re_deformed_subset_center_x= x_pos(i) -search_radius : x_pos(i) + search_radius 
                 % 以该点为中点，构造变形子区,只是平移
                    for m = re_deformed_subset_center_y - r : re_deformed_subset_center_y + r  
                         for n = re_deformed_subset_center_x - r : re_deformed_subset_center_x +r
                                patch_m = m - re_deformed_subset_center_y +r + 1 ;
                                patch_n = n - re_deformed_subset_center_x +r + 1 ;
                                gray_level = image_deformed(m,n);
                                deformed_subset(patch_m,patch_n) = gray_level; 
                            
                         end
                     end    
                        
                     correlation_coefficient = Calc_correlation_coefficient(refer_subset,deformed_subset);%相关系数计算
                      
                     index_k = re_deformed_subset_center_y - y_pos(i) + search_radius + 1;
                     index_l = re_deformed_subset_center_x - x_pos(i) + search_radius + 1;
                        
                     correlation_coefficient_mat_per_point(index_k,index_l) = correlation_coefficient;
                    
                 end
            end 
               
         
            [y,x]=find(correlation_coefficient_mat_per_point==min(min(correlation_coefficient_mat_per_point)));
          
            resultx=x_pos(i)-search_radius+x-1;
            resulty=y_pos(i)-search_radius+y-1;
            result_matx(i)=resultx;
            result_maty(i)=resulty;
            
            %用以上结果作为sub-pixel计算的初值估计，用ic_gn方法做亚像素计算
            U= resultx-x_pos(i);
            V= resulty-y_pos(i);
            
            sub_u_result_matx(i)=U;
            sub_v_result_maty(i)=V;
            
            %IC_GN
            P=[U,0,0,V,0,0].'; %迭代形函数参数初值
            middle_mat=Middle_mat(refer_subset,r); %计算参考子区雅各比矩阵
 
            for IT_count=1:20
                interpolation_deform_subset=Interpolation_deform(image_deformed,P,x_pos(i),y_pos(i),r);%插值计算目标子区灰度
                [delta_P,P_next]=IC_GN2(middle_mat,refer_subset,interpolation_deform_subset,r,P);
                P=P_next;
                norm=sqrt(delta_P(1)^2+(delta_P(2)*r)^2+(delta_P(3)*r)^2+delta_P(4)^2+(delta_P(5)*r)^2+(delta_P(6)*r)^2);%P的范数
                fprintf('%d , %d, %f\n',i,IT_count,norm);
               if norm<=0.01                    
                   break
               end
               
               if ((P(1)-U)^2+(P(4)-V)^2)>100    
                   bug(bi)=i;
                   bi=bi+1;
                   break
                end
               %判断，偏离整像素点太多;
            end
            
            if (IT_count>400)
                err(ei)=i;
                ei=ei+1;
            end
            sub_u_result_matx(i)=P(1);
            sub_v_result_maty(i)=P(4);
            
            %画出计算结果位置
    
            x_posnew(i) =x_pos1(i)+sub_u_result_matx(i);
            y_posnew(i) =y_pos1(i)+sub_v_result_maty(i);
            plot(x_posnew(i),y_posnew(i),'o','Color','y');
            plot(x_posnew(i),y_posnew(i),'.','Color','r');
            hold on
            
    end   



    
    %{
for i=1:POI_point_count      %画出计算结果位置
    
            x_posnew(i) =x_pos1(i)+sub_u_result_matx(i);
            y_posnew(i) =y_pos1(i)+sub_v_result_maty(i);
            plot(x_posnew(i),y_posnew(i),'o','Color','y');
            hold on
    end
    for j=1:POI_point_count 
            text(x_posnew(j),y_posnew(j),num2str(j));
    end
    %}
centersnew=zeros(POI_point_count,2)*NaN;
centersnew(:,1)=x_posnew(:)';
centersnew(:,2)=y_posnew(:)';
filename1   =   strcat('centersnew',num2str(count),'.csv');
csvwrite(filename1, centersnew);
filename1   =   strcat('centersnew',num2str(count),'.mat');
save(filename1, 'centersnew');
