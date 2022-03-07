% Load first n sample for now, n must be even
firstN = 300;
bl = 1; bu = 1024; % lower and upper for image clamp

% Paths
D1 = 'C:\Irmak Dosyalar\HW\5.1\CS 551\Project\images';
D2 = 'C:\Irmak Dosyalar\HW\5.1\CS 551\Project\labels';

% Read images
S1 = dir(fullfile(D1,'*.png'));
S2 = dir(fullfile(D2,'*.json'));

for k = 1:firstN
    F1 = fullfile(D1,S1(k).name); I = imread(F1);
    F2 = fullfile(D2,S2(k).name); fid = fopen(F2);

    if(~isempty(strfind(S2(k).name,'post')))

        raw = fread(fid,inf); str = char(raw'); 
        fclose(fid); val = jsondecode(str);

        for i=1:length(val.features.xy)
            exp =  '[^()]*';
            str = val.features.xy(i).wkt;
            nums = regexp(str, '(?<=\()[^)]*(?=\))', 'match', 'once');
            nums = nums(2:length(nums));
            nums = split(nums,",");
            coordinates = split(strtrim(split(nums,","))," ");
            coordinates = min(max(round(str2double(coordinates)),bl),bu);
            [mins,i_mins] = min(coordinates);
            [maxs,i_maxs] = max(coordinates);
            
            damagelabel = val.features.lng_lat(i).properties.subtype;
            folder = 'D://Try//';
            folder = strcat(folder,damagelabel);
            
            baseFileName = split(val.metadata.img_name,".");
            baseFileName = strcat(baseFileName(1),"_",int2str(i),".png");
            fullFileName = fullfile(folder, baseFileName);
            imwrite(I(mins(2):maxs(2),mins(1):maxs(1),:), fullFileName);
        end
    end
end


