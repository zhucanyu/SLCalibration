function interpolation_deform_mat=Interpolation_deform(image_deformed,P,x_pos,y_pos,r)
%插值计算目标子区灰度
interpolation_deform_mat=zeros(2*r+1,2*r+1);
for patch_m=1:2*r+1
    for patch_n=1:2*r+1
        x=patch_n-(r+1);             %计算参考子区对应目标子区像素点图像坐标x,y_new
        y=patch_m-(r+1);
        x_1=(P(2)+1)*x+P(3)*y+P(1);
        y_1=P(5)*x+(P(6)+1)*y+P(4);
        x_new=x_pos+x_1;
        y_new=y_pos+y_1;
        m=floor(y_new);  %找到最近的整数点
        n=floor(x_new);
        around=zeros(6,6);%取点周围的6*6矩阵
        around=double(image_deformed(m-2:m+3,n-2:n+3));
        %插值得到灰度
        d_y=y_new-m;
        d_x=x_new-n;
        
        Vx=[d_x^3;d_x^2;d_x;1];
        Vy=[d_y^3,d_y^2,d_y,1];
        Q=[  1/11  ,  -6/11  ,  13/11  , -13/11  ,  6/11  ,-1/11;
           -45/209 , 270/209 ,-453/209 , 288/209 ,-72/209 ,12/209;
           26/209  ,-156/209 , -3/209  , 168/209 ,-42/209 , 7/209;
           0       ,  0      ,    1    ,   0     ,   0    , 0     ];
        gray=Vy*Q*around*Q'*Vx;
        
        %gray=interp2(1:6,1:6,around,3+d_x,3+d_y,'spline');
        
        
        interpolation_deform_mat(patch_m,patch_n)=gray;
    end
end

end

