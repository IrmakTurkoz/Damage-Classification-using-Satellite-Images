bl = 1; bu = 1024; % lower and upper for image clamp
sizex = 112; sizey = 112;
% Paths
D1 = 'C:\Xview2\train\train\images';
D2 = 'C:\Xview2\train\train\labels';

% Read images
S1 = dir(fullfile(D1,'*.png'));
S2 = dir(fullfile(D2,'*.json'));

for k = 1:length(S1)
    disp(k);
    F1 = fullfile(D1,S1(k).name); I = imread(F1);
    F2 = fullfile(D2,S2(k).name); fid = fopen(F2);

    if(~isempty(strfind(S2(k).name,'post')))
        IPost = I;
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
            
            me = max(sizex+1,min(round(mean(coordinates)), 1024-sizex-1));
            
            damagelabel = val.features.lng_lat(i).properties.subtype;
            distype = val.metadata.disaster_type;

            if(strcmp(damagelabel,'un-classified')) 
                continue;
            end
            
            folder = 'C://LastPart//Post//';
            folder = strcat(folder, distype);
            folder = strcat(folder, '//');
            folder = strcat(folder,damagelabel);
            
            baseFileName = split(val.metadata.img_name,".");
            baseFileName = strcat(baseFileName(1),"_",int2str(i),".png");
            fullFileName = fullfile(folder, baseFileName);
            imwrite(I(me(2)-sizey:me(2)+sizey-1,me(1)-sizex:me(1)+sizey-1,:), fullFileName);
        end
    else
        fclose(fid);
        % This is pre file
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
            
            me = max(sizex+1,min(round(mean(coordinates)), 1024-sizex-1));
            
            damagelabel = val.features.lng_lat(i).properties.subtype;
            distype = val.metadata.disaster_type;

            if(strcmp(damagelabel,'un-classified')) 
                continue;
            end
            
            folder = 'C://LastPart//Pre//';
            folder = strcat(folder, distype);
            folder = strcat(folder, '//');
            folder = strcat(folder,damagelabel);
            
            folder2 = 'C://LastPart//Diff//';
            folder2 = strcat(folder2, distype);
            folder2 = strcat(folder2, '//');
            folder2 = strcat(folder2,damagelabel);

            baseFileName = split(val.metadata.img_name,".");
            baseFileName = strcat(baseFileName(1),"_",int2str(i),"_pre.png");
            fullFileName = fullfile(folder, baseFileName);
            
            fullFileNameDiff = fullfile(folder2, baseFileName);

            %For grayscale
            %IGray = uint8(filter2(fspecial('gaussian'), rgb2gray(I)));
            %IPostGray = uint8(filter2(fspecial('gaussian'), rgb2gray(IPost)));
            %IDiff = imabsdiff(IGray, IPostGray);
            
            %For color
            IDiff = imabsdiff(I, IPost);
            
            imwrite(I(me(2)-sizey:me(2)+sizey-1,me(1)-sizex:me(1)+sizey-1,:), fullFileName);
            imwrite(IDiff(me(2)-sizey:me(2)+sizey-1,me(1)-sizex:me(1)+sizey-1,:), fullFileNameDiff);

        end
    end
end