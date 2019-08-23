clc; clear all;

addpath('/usr/local/MATLAB/R2016b/toolbox/jsonlab-1.5')

standard_size = [360,640];

attr = 'test'; %'train' or 'test'
attr_cate = [attr,'_data'];
main_path = '/home/zhouzhp/C-3-Framework-python3.x/datasets/ProcessedData/FDST/';
read_main_path = [main_path,attr_cate];
output_path = [main_path,attr,'/','img','/'];
output_path_den = [main_path,attr,'/','den','/'];

dirs = dir(read_main_path);

len = length(dirs);


mkdir(output_path);

for idx=28:len  
    idx_char = num2str(dirs(idx).name);
    idx_int = str2num(dirs(idx).name);
    read_sub_path = [read_main_path,'/',idx_char]; %/traindata/1
    for k = 1:150
        fprintf(1,'Processing %d_/%d files\n', idx_int, k);
        input_image_name = [read_sub_path,'/',num2str(k,'%03d'),'.jpg'];
        input_jason_name = [read_sub_path,'/',num2str(k,'%03d'),'.json'];
        im = imread(input_image_name);  
        jason_file = loadjson(input_jason_name);
        s2c = struct2cell(jason_file(1));
        regions = s2c{1,1}.regions;
        annPoints = rect2point(regions);
        [h, w, c] = size(im);
        rate = standard_size(1)/h;
        rate_w = w*rate;
        if rate_w>standard_size(2)
            rate = standard_size(2)/w;
        end
        rate_h = double(int16(h*rate))/h;
        rate_w = double(int16(w*rate))/w;
        im = imresize(im,[int16(h*rate),int16(w*rate)]);
        annPoints(:,1) = annPoints(:,1)*double(rate_w);
        annPoints(:,2) = annPoints(:,2)*double(rate_h);
%     
        im_density = get_density_map_gaussian(im,annPoints,15,4); 
        im_density = im_density(:,:,1);
        imwrite(im, [output_path,idx_char,'_',num2str(k) '.jpg']);
        csvwrite([output_path_den ,idx_char,'_',num2str(k),'.csv'], im_density);
        
        
        
        
    end
end



% output_path = '/home/zhouzhp/C-3-Framework-python3.x/datasets/ProcessedData/mall_dataset/';
% img_path = '/home/zhouzhp/C-3-Framework-python3.x/datasets/ProcessedData/mall_dataset/frames/';
% processed_img_path = '/home/zhouzhp/C-3-Framework-python3.x/datasets/ProcessedData/mall_dataset/processed_img/';
% load('/home/zhouzhp/C-3-Framework-python3.x/datasets/ProcessedData/mall_dataset/mall_gt.mat');
% den_path = '/home/zhouzhp/C-3-Framework-python3.x/datasets/ProcessedData/mall_dataset/dense/';
% 
% mkdir(den_path);
% mkdir(processed_img_path);
% 
% num_images = 2000;
% 
% for idx = 1:num_images
%     i = idx;
%     if (mod(idx,10)==0)
%         fprintf(1,'Processing %3d/%d files\n', idx, num_images);
%     end
%     input_img_name = strcat(img_path,'seq_',num2str(i,'%06d'),'.jpg');
%     im = imread(input_img_name);  
%     [h, w, c] = size(im);
%     annPoints =  frame{i}.loc;
% 
% 
%     rate = standard_size(1)/h;
%     rate_w = w*rate;
%     if rate_w>standard_size(2)
%         rate = standard_size(2)/w;
%     end
%     rate_h = double(int16(h*rate))/h;
%     rate_w = double(int16(w*rate))/w;
%     im = imresize(im,[int16(h*rate),int16(w*rate)]);
%     annPoints(:,1) = annPoints(:,1)*double(rate_w);
%     annPoints(:,2) = annPoints(:,2)*double(rate_h);
%     
%     im_density = get_density_map_gaussian(im,annPoints,15,4); 
%     im_density = im_density(:,:,1);
%     
%     imwrite(im, [processed_img_path num2str(idx) '.jpg']);
%     csvwrite([den_path num2str(idx) '.csv'], im_density);
% end