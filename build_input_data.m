clear
close all
dataRoot = '..\raw_data\';
% dateString = '20211028';
dateString = '20211124';
dataFolder = [dataRoot dateString filesep];

% load in table with region categories
image_key = readtable([dataRoot 'image_key.csv']);
% image_key = readtable('20211124\image_key.csv');

% specify granularity to use for chopping samples into finer grids
snip_size = 176;
scale_factor = 224/snip_size;

% grain_size_cell = {'500_','5001','12mm','_2mm'};
grain_size_cell = {'sand_'};
for g = 1:length(grain_size_cell)
    grain_size = grain_size_cell{g};
    % specify aggregate size to use
    snip_size_string = [num2str(grain_size) '_' num2str(snip_size)];
    snip_size_string = strrep(snip_size_string,'__','_');
    file_list = dir([dataFolder '*' num2str(grain_size) '*jpg']);
    
    % extract key info
    valid_name_cell = image_key.string;
    region_string_cell = image_key.region;
    region_id_cell = image_key.region_code;
    
    % set writepath 
    OutPath = ['..\built_data\' dateString filesep snip_size_string filesep];
    mkdir(OutPath)
    wb = waitbar(0,'generating image snips...');
    for f = 1:length(file_list)
        waitbar(f/length(file_list),wb);
        fname = file_list(f).name;
        
        % extract basic image ID info
        underscores = strfind(fname,'_');
        jpg = strfind(fname,'.JPG');    
        source_string_2 = fname(underscores(1)+1:underscores(2)-1);
            
        v_flags = contains(valid_name_cell,source_string_2);
        if any(v_flags)
            
            source_string = region_string_cell{v_flags};%fname(1:underscores(1)-1);
            sample_num = str2num(fname(underscores(end)+1:jpg-1));
    
            % make directory to store image snips
            savePath = [OutPath source_string filesep];
            mkdir(savePath);
    
            % load in image
            im_raw_bg = imread([dataFolder fname]);
            sz_vec = size(im_raw_bg(:,:,1));
    
            % get circle diam
            circ_rad = floor(0.42*min(size(im_raw_bg(:,:,1))));
    
            % find image center
            ct = ceil(size(im_raw_bg(:,:,1))/2);
    
            % calculate sides of circumscribed square
            sq_len = sind(45)*circ_rad;
            n_reps_raw = 2*sq_len/snip_size;    
            n_reps = floor(n_reps_raw);
            sz_temp = floor(2*sq_len/n_reps);
    
            % find bottom left corner
            start_coord = uint16(ceil(ct-sq_len));
    
            % generate indexing arrays
            row_array = zeros(sz_vec);
            col_array = row_array;
            row_vec = repelem(1:n_reps,sz_temp)';
            row_array(start_coord(1):start_coord(1)+length(row_vec)-1,:) = repmat(row_vec,1,sz_vec(2));
            col_array(:,start_coord(2):start_coord(2)+length(row_vec)-1,:) = repmat(row_vec',sz_vec(1),1);
    
            % generate snips
            snip_cell = cell(1,n_reps^2);
            for i = 1:n_reps     
                for j = 1:n_reps
                    ind = (i-1)*n_reps + j;
                    logic_array = row_array==i&col_array==j;
                    i_temp = zeros(sz_temp,sz_temp,3,'uint8');
                    for k = 1:3
                        slice = im_raw_bg(:,:,k);
                        i_temp(:,:,k) = reshape(slice(logic_array),sz_temp,sz_temp);
                    end
                    snip_cell{ind} = imresize(i_temp,snip_size/sz_temp);
                end
            end    
    
            % save snips 
            for s = 1:length(snip_cell)
                save_string = ['im_' source_string_2 '_' sprintf('%03d',sample_num) '_' sprintf('%03d',s) '.jpg'];
                snip = snip_cell{s};
                snip_rs = imresize(snip,scale_factor);
                imwrite(snip_rs,[savePath save_string]);
            end
        end
    end   
    delete(wb);
end    