function [mean] = findMeanDamageLevel(sum,n)
mean = 0;
ratio = sum / n;
if ratio <= 1
    mean = 1;
elseif ratio <= 2
    mean = 2;
elseif ratio <= 3
    mean = 3;
else 
    mean = 4;
end
end

