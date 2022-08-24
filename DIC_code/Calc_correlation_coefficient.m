function [ correlation_coefficient ] = Calc_correlation_coefficient( refer_subset,deformed_subset )
% Calc_correlation_coefficient 计算相关系数
[m,n] = size(refer_subset);
aver_refer_subset = sum(sum(refer_subset))*1.0 / m/ n;
aver_deformed_subset = sum(sum(deformed_subset))*1.0 / m/ n;
refer_subset_minus_aver = refer_subset - aver_refer_subset;
deformed_subset_minus_aver = deformed_subset - aver_deformed_subset ;
sqrt_sum_refer_subset_minus_aver = sqrt(double(sum(sum(refer_subset_minus_aver.*refer_subset_minus_aver))));
sqrt_sum_deformed_subset_minus_aver = sqrt(double(sum(sum(deformed_subset_minus_aver.*deformed_subset_minus_aver))));

% sqrt_sum_refer_subset_minus_aver = sqrt(double(sum(sum(refer_subset_minus_aver))) - aver_refer_subset);
% sqrt_sum_deformed_subset_minus_aver = sqrt(double(sum(sum(deformed_subset_minus_aver))) - aver_deformed_subset);

new_subset = refer_subset_minus_aver/sqrt_sum_refer_subset_minus_aver - deformed_subset_minus_aver/sqrt_sum_deformed_subset_minus_aver ;

correlation_coefficient = sum(sum(new_subset.*new_subset));
end

