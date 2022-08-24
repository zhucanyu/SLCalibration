function [middle_mat] = Middle_mat(refer_subset,r)
%求计算deltaP左边的矩阵 -（J^T*J）^-1*J^T 输入 返回此矩阵
%   此处显示详细说明


%求梯度矩阵
gra_x=zeros(2*r+1,2*r+1);
gra_y=zeros(2*r+1,2*r+1);
%逐行求x方向梯度矩阵
for m=1:2*r+1
    row=refer_subset(m,:);
    gra=gradient(row,2*r+1);
    gra_x(m,:)=gra;
end
%逐列求y方向梯度矩阵
for n=1:2*r+1
    col=refer_subset(:,n);
    gra=gradient(col,2*r+1);
    gra_y(:,n)=gra;
end
i=1;


for patch_m=1:2*r+1   
    for patch_n=1:2*r+1
        G=[gra_x(patch_m,patch_n),gra_y(patch_m,patch_n)]; 
        x=patch_n-(r+1);
        y=patch_m-(r+1);
        J(i,:)=G*[1,x,y,0,0,0;0,0,0,1,x,y];
        %hessian_mat=hessian_mat+(G*[1,x,y,0,0,0;0,0,0,1,x,y])'*(G*[1,x,y,0,0,0;0,0,0,1,x,y]);
        i=i+1;
    end
end




% 计算Hessian矩阵


middle_mat=-inv((J'*J))*J';



end

