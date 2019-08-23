function  annPoints = rect2point( regions )
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
tmp = size(regions);
annPoints = rand(tmp(2),2);
s = tmp(2);
for i=1:s
    x = regions{1,i}.shape_attributes.x;
    y = regions{1,i}.shape_attributes.y;
    width = regions{1,i}.shape_attributes.width;
    height = regions{1,i}.shape_attributes.height;
    center_x = x + width / 2.0;
    center_y = y + height / 2.0;
    annPoints(i,1) = center_x;
    annPoints(i,2) = center_y;
end
end

