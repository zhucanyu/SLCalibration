clc;
clear;
close all;

R = csvread('R.csv'); %读入外参数据

[U12,S12,V12] = svd(R);
R = U12*V12';
T = csvread('T.csv');

mtxl = csvread('mtxl.csv');%读入左右相机内参
mtxr = csvread('mtxr.csv');
distl = csvread('distl.csv');
distr = csvread('distr.csv');

%读入标定板与相机关系矩阵

centers1=csvread('centers1.csv');


fxl= mtxl(1,1);  %opencv畸变模型左相机参数
 fyl = mtxl(2,2);
 fsl = 0;
 cxl = mtxl(1,3);
 cyl = mtxl(2,3);
 k1l = distl(1);
 k2l = distl(2);
 p1l = distl(3);
 p2l = distl(4);
 k3l = distl(5);

 fxr = mtxr(1,1); %opencv畸变模型右相机参数
 fyr = mtxr(2,2);
 fsr = 0;
 cxr = mtxr(1,3);
 cyr = mtxr(2,3);
 k1r = distr(1);
 k2r = distr(2);
 p1r = distr(3);
 p2r = distr(4);
 k3r = distr(5);

POI_point_count = size(centers1,1); %点数量
centers1=0;

for count=1:5
   
    centers1_1=zeros(POI_point_count,2)*NaN;%定义左图的去畸变矩阵
    centersnew1_1=zeros(POI_point_count,2)*NaN;%定义右图的去畸变矩阵
    
    filename1   =   strcat('rl',num2str(count),'.mat');
    filename2   =   strcat('centers',num2str(count),'.csv');
    filename3   =   strcat('centersnew',num2str(count),'.csv');
    filename4 = strcat('tvecsl',num2str(count),'.csv');
    centers1=csvread(filename2); %读入左右图片点坐标
   
    centersnew1=csvread(filename3);
    
    
   %  centers1_1=centers1;
   % centersnew1_1=centersnew1;

for i=1: POI_point_count %左图opencv去畸变
				 y11 = (centers1(i,2) - cyl) / fyl;
				 x11 = (centers1(i,1) - cxl - fsl*y11) / fxl;
				 r2 = x11*x11 + y11*y11;
				 x1 = (x11 - 2 * p1l*x11*y11 - p2l*(r2 + 2 * x11*x11))/ (1 + k1l*r2 + k2l*r2*r2 + k3l*r2*r2*r2);
				 y1 = (y11 - 2 * p2l*x11*y11 - p1l*(r2 + 2 * y11*y11))/ (1 + k1l*r2 + k2l*r2*r2 + k3l*r2*r2*r2);
                %x1 = x11*(1 + k1l*r2 + k2l*r2*r2 + k3l*r2*r2*r2)+2 * p1l*x11*y11 + p2l*(r2 + 2 * x11*x11);
                %y1 = y11*(1 + k1l*r2 + k2l*r2*r2 + k3l*r2*r2*r2)+2 * p2l*x11*y11 + p1l*(r2 + 2 * y11*y11);
				centers1_1(i,1) = cxl + fxl*x1+ fsl*y1;
				centers1_1(i,2) = cyl + fyl*y1;   
end

    for k=1:50 %迭代次数
	      for i=1: POI_point_count
	
				 y11 = (centers1_1(i,2) - cyl) / fyl;
				 x11 = (centers1_1(i,1) - cxl - fsl*y11) / fxl;
				 r2 = x11*x11 + y11*y11;
                 x1 = ((centers1(i,1) - cxl - fsl*y11) / fxl - 2 * p1l*x11*y11 - p2l*(r2 + 2 * x11*x11))/ (1 + k1l*r2 + k2l*r2*r2 + k3l*r2*r2*r2);
				 y1 = ((centers1(i,2) - cyl) / fyl - 2 * p2l*x11*y11 - p1l*(r2 + 2 * y11*y11))/ (1 + k1l*r2 + k2l*r2*r2 + k3l*r2*r2*r2);
                 
				 %x1 = ((centers1(i,1) - cxl - fsl*y11) / fxl)*(1 + k1l*r2 + k2l*r2*r2 + k3l*r2*r2*r2)+2 * p1l*x11*y11 + p2l*(r2 + 2 * x11*x11);
                 %y1 = ((centers1(i,2) - cyl) / fyl)*(1 + k1l*r2 + k2l*r2*r2 + k3l*r2*r2*r2)+2 * p2l*x11*y11 + p1l*(r2 + 2 * y11*y11);
				centers1_1(i,1) = cxl + fxl*x1+ fsl*y1;
				centers1_1(i,2) = cyl + fyl*y1;   
          end  
    end

for i=1: POI_point_count %you图opencv去畸变
				 y11 = (centersnew1(i,2) - cyr) / fyr;
				 x11 = (centersnew1(i,1) - cxr - fsr*y11) / fxr;
				 r2 = x11*x11 + y11*y11;
				 x1 = (x11 - 2 * p1r*x11*y11 - p2r*(r2 + 2 * x11*x11))/ (1 + k1r*r2 + k2r*r2*r2 + k3r*r2*r2*r2);
				 y1 = (y11 - 2 * p2l*x11*y11 - p1r*(r2 + 2 * y11*y11))/ (1 + k1r*r2 + k2r*r2*r2 + k3r*r2*r2*r2);
				centersnew1_1(i,1) = cxr + fxr*x1+ fsr*y1;
				centersnew1_1(i,2) = cyr + fyr*y1;   
end

for k=1:50 %迭代次数
	      for i=1: POI_point_count
	
				 y11 = (centersnew1_1(i,2) - cyr) / fyr;
				 x11 = (centersnew1_1(i,1) - cxr - fsr*y11) / fxr;
				 r2 = x11*x11 + y11*y11;
				 x1 = ((centersnew1(i,1) - cxr - fsr*y11) / fxr - 2 * p1r*x11*y11 - p2r*(r2 + 2 * x11*x11))/ (1 + k1r*r2 + k2r*r2*r2 + k3r*r2*r2*r2);
				 y1 = ((centersnew1(i,2) - cyr) / fyr - 2 * p2l*x11*y11 - p1r*(r2 + 2 * y11*y11))/ (1 + k1r*r2 + k2r*r2*r2 + k3r*r2*r2*r2);
				centersnew1_1(i,1) = cxr + fxr*x1+ fsr*y1;
				centersnew1_1(i,2) = cyr + fyr*y1;   
          end  
 end
  
%三维重建    
centers1w= zeros(POI_point_count,3)*NaN*NaN; %定义世界坐标矩阵
P=zeros(4,3)*NaN; %最小二乘所需矩阵
Q=zeros(4,1)*NaN;
M=zeros(3,1)*NaN;

for i=1: POI_point_count %求三维坐标
  
P(1,1)=-fxl;
P(1,2)=-fsl;
P(1,3)= centers1_1(i,1)-cxl;
P(2,1)=0;
P(2,2)=-fyl;
P(2,3)=centers1_1(i,2)-cyl;

P(3,1)=(centersnew1_1(i,1)-cxr)*R(3,1)-fxr*R(1,1)-fsr*R(2,1);
P(3,2)=(centersnew1_1(i,1)-cxr)*R(3,2)-fxr*R(1,2)-fsr*R(2,2);
P(3,3)=(centersnew1_1(i,1)-cxr)*R(3,3)-fxr*R(1,3)-fsr*R(2,3);
P(4,1)=(centersnew1_1(i,2)-cyr)*R(3,1)-fyr*R(2,1);
P(4,2)=(centersnew1_1(i,2)-cyr)*R(3,2)-fyr*R(2,2);
P(4,3)=(centersnew1_1(i,2)-cyr)*R(3,3)-fyr*R(2,3);
Q(1)=0;
Q(2)=0;
Q(3)=fxr*T(1)+fsr*T(2)-(centersnew1_1(i,1)-cxr)*T(3);
Q(4)=fyr*T(2)-(centersnew1_1(i,2)-cyr)*T(3);
M=inv(P'*P)*P'*Q; %最小二乘
centers1w(i,1)=M(1);
centers1w(i,2)=M(2);
centers1w(i,3)=M(3);
end

centers1w(:,3)=centers1w(:,3)-mean(centers1w(:,3));
centers1w(:,3)=centers1w(:,3)-min(centers1w(:,3));
centers1w(:,2)=centers1w(:,2)-mean(centers1w(:,2));
centers1w(:,1)=centers1w(:,1)-mean(centers1w(:,1));

%centers1w_b = (Ry*(centers1w_b'))';
%figure (1);
%subplot(2,3,count);plot3(centers1w_b(:,1),centers1w_b(:,2),centers1w_b(:,3),'r.');
%hold on
figure (2);
subplot(2,3,count);
plot3(centers1w(:,1),centers1w(:,2),centers1w(:,3),'r.');
csvwrite(strcat('cw',num2str(count),'.csv'), centers1w);

hold on
%axis equal
end
	