%Customizing matlab
clear;
clc;
close all;
warning('off','all')
warning
set(0,'DefaultFigureWindowStyle','docked')

% Load first n sample for now, n must be even
firstN = 259;

% Paths
D1 = 'images';
D2 = 'labels';

% Read images and jsons
image_dir = dir(fullfile(D1,'*.png'));
file_size = length(image_dir);
label_dir = dir(fullfile(D2,'*.json'));

dmg_levels = ["no-damage", "minor-damage", "major-damage", "destroyed"];

F_label = fullfile(D2,label_dir.name);
data_labels = cell(4,firstN);
flag = true;
damage_i = 1;
index = 1;
count_firstN = ones(1,4);
data = cell(4,firstN);
while flag
    if index > length(label_dir)
        flag = false;
    end
    current_file = label_dir(index).name;
    fid = fopen(current_file);
    raw = fread(fid,inf);
    str = char(raw');
    fclose(fid);
    jsons = jsondecode(str);
    sum_dmg = 0;
    if contains(current_file,'post')
        for j = 1 : length(jsons.features.lng_lat)
            
            if (isfield(jsons.features.lng_lat(j).properties,'subtype'))
                x =  (strfind (dmg_levels, jsons.features.lng_lat(j).properties.subtype));
                sum_dmg = sum_dmg +find(~cellfun(@isempty,x));
            end
        end
        if  not(isempty(sum_dmg))&& not(isempty(j)) && j> 0
            dmg = findMeanDamageLevel(sum_dmg,j);
            if count_firstN(dmg) <= firstN
                
                p = fullfile(D1,image_dir(index+1).name);
                I  = cell2mat({imread(p)});
                pre = imresize(I, 0.25);
                p = fullfile(D1,image_dir(index).name);
                I  = cell2mat({imread(p)});
                post = imresize(I, 0.25);
                
                abs_difference = pre-post;
                
                data{dmg,count_firstN(dmg)} = abs_difference;
                
%                 figure;
%                 imshow(abs_difference);
%                 drawnow;
%                 pause(2);
                
                count_firstN(dmg) = count_firstN(dmg) + 1;
            end
        end
        
        if all(count_firstN(:) > firstN) || index >= length(label_dir)
            flag = false;
            
        end
    end
    index = index + 1;
    
    
end


