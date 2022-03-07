
%Customizing matlab
clear;
clc;
close all;
warning('off','all')
warning
set(0,'DefaultFigureWindowStyle','docked')

% Load first n sample for now, n must be even
firstN = 100;

Posts = zeros(1024,1024,3,firstN);
Pres = zeros(1024,1024,3,firstN);
Labels = strings(firstN,1);

% Paths
D1 = 'images';
D2 = 'labels';


% Read images and jsons
image_dir = dir(fullfile(D1,'*.png'));
file_size = length(image_dir);
label_dir = dir(fullfile(D2,'*.json'));

file_names = string({image_dir.name});
file_names = horzcat(file_names);
l_index = 1;
for k = 1:firstN
    finded_files = strfind(file_names(:),strcat('', sprintf('%08d',k-1)));
    finded_files = find(~cellfun(@isempty,finded_files));
    
    
    for j = 1: length(finded_files)
        F = fullfile(D1,image_dir(finded_files(j)).name);
        %disp("Now reading " + image_dir(finded_files(j)).name);

        I = imread(F);
        F_l = fullfile(D2,label_dir(finded_files(j)).name);
        fid = fopen(F_l);
        raw = fread(fid,inf);
        str = char(raw');
        fclose(fid);
        val = jsondecode(str);
        
        if(strfind(image_dir(finded_files(j)).name,strcat('', 'pre')))
            Posts(:,:,:,ceil(k/2)) = I;
            Posts(:,:,:,ceil(k/2)) = Posts(:,:,:,ceil(k/2)) ./ 255;
            Labels(l_index) = val.metadata.disaster_type;
            %disp("" + Labels(l_index));
            l_index = l_index+1;
        else
            Pres(:,:,:,ceil(k/2)) = I;
            Pres(:,:,:,ceil(k/2)) = Pres(:,:,:,ceil(k/2)) ./ 255;
        end
        
        %imshow(I);
        %drawnow;
    end
end

