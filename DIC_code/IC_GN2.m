function [delta_P,P_next] = IC_GN2(middle_mat,refer_subset,interpolation_deform_subset,r,P)
%IC_GN方法，由hessian矩阵，目标子区灰度梯度gra_x,gra_y，求delta_P,返回delta_P，n+1次迭代的P_next
%左边矩阵-（J^T*J）^-1*J^T已知，为middle_mat

%计算F矩阵
F=zeros((2*r+1)^2,1);
delta_fg=0;
delta_g2=0;
delta_f2=0;
avg_f=sum(sum(refer_subset))/(2*r+1)^2;
avg_g=sum(sum(interpolation_deform_subset))/(2*r+1)^2;
for m=1:2*r+1
    for n=1:2*r+1
        delta_fg= delta_fg+(refer_subset(m,n)-avg_f)*(interpolation_deform_subset(m,n)-avg_g);
        delta_g2= delta_g2+(interpolation_deform_subset(m,n)-avg_g)^2;
        delta_f2= delta_f2+(refer_subset(m,n)-avg_f)^2;
    end
end
i=1;
delta_g=delta_g2^0.5;
delta_f=delta_f2^0.5;
%F=zeros(6,1);
for m=1:2*r+1
    for n=1:2*r+1      
        F(i)= delta_g/delta_f*(refer_subset(m,n)-avg_f)-(interpolation_deform_subset(m,n)-avg_g) ;
        %x=n-(r+1);
        %y=m-(r+1);
        %F(i)=((refer_subset(m,n)-avg_f)-delta_f/delta_g*(interpolation_deform_subset(m,n)-avg_g)) ;
        i=i+1;
    end
end
%相乘得到delta_P
delta_P=(middle_mat)*F;
%组合得到迭代形函数
W=[P(2)+1,P(3),P(1);P(5),P(6)+1,P(4);0,0,1]/[delta_P(2)+1,delta_P(3),delta_P(1);delta_P(5),delta_P(6)+1,delta_P(4);0,0,1];
%得到参数P_next
P_next=[W(1,3),W(1,1)-1,W(1,2),W(2,3),W(2,1),W(2,2)-1];
end

