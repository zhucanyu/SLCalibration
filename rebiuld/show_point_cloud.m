clc
clear 
close all

points=csvread('pc1.csv');

points(:,3)=points(:,3)-mean(points(:,3));
points(:,3)=points(:,3)-min(points(:,3));
points(:,2)=points(:,2)-mean(points(:,2));
points(:,1)=points(:,1)-mean(points(:,1));

xmax=round(max(points(:,1)));xmin=round(min(points(:,1)));
ymax=round(max(points(:,2)));ymin=round(min(points(:,2)));
[X,Y]=meshgrid(xmin:(xmax-xmin)/100:xmax,ymin:(ymax-ymin)/100:ymax);
Z=griddata(points(:,1),points(:,2),points(:,3),X,Y);

figure(1)
mesh(X,Y,Z)
axis equal
figure(2)
light_from=[1,0,1];
surfl(X,Y,Z,light_from,'light');
shading interp
xlabel('X')
ylabel('Y')
axis equal