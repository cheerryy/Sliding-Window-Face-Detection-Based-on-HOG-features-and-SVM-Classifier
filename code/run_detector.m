% Starter code prepared by James Hays for CS 143, Brown University
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confidences, image_ids] = .... 
    run_detector(test_scn_path, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
 
% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')
 
% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.
 
% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.
 
test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));
 
%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);
iter = 0.8;
scale_vector = [1,iter,iter^2,iter^3,iter^4,iter^5,iter^6];
threshold = 0.75;
 
for i = 1:length(test_scenes)
      
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    
    cur_bboxes = zeros(0,4);
    cur_confidences = zeros(0,1);
    cur_image_ids = cell(0,1);
    
    for scale = scale_vector
        scaled_img = imresize(img,scale);
        [height,width] = size(scaled_img);
        HOG = vl_hog(scaled_img,feature_params.hog_cell_size);
        window_x_limit = floor(width/feature_params.hog_cell_size) - ...
            feature_params.template_size/feature_params.hog_cell_size + 1;
        window_y_limit = floor(height/feature_params.hog_cell_size) - ...
            feature_params.template_size/feature_params.hog_cell_size + 1;
        
        D = (feature_params.template_size/feature_params.hog_cell_size)^2*31;
        features = zeros(window_x_limit * window_y_limit,D);
        for x=1:window_x_limit
            for y=1:window_y_limit
                features((x-1)*window_y_limit+y,:)=reshape(HOG(y:(y+feature_params.template_size/feature_params.hog_cell_size-1),...
                    x:(x+feature_params.template_size/feature_params.hog_cell_size-1),:),...
                    [1,D]);
            end
        end
        scores = features * w + b;
        indices = find(scores>threshold);
        scale_confidences = scores(indices);
        
        x_hog = floor(indices./ window_y_limit);
        y_hog = mod(indices,window_y_limit)-1;
        scale_x_min = x_hog*feature_params.hog_cell_size+1;
        scale_x_max = x_hog*feature_params.hog_cell_size+feature_params.template_size;
        scale_y_min = y_hog*feature_params.hog_cell_size+1;
        scale_y_max = y_hog*feature_params.hog_cell_size+feature_params.template_size;
        scale_bboxes = [scale_x_min, scale_y_min, scale_x_max, scale_y_max]./scale;
        scale_image_ids = repmat({test_scenes(i).name}, size(indices,1), 1);
        
        cur_bboxes      = [cur_bboxes;      scale_bboxes];
        cur_confidences = [cur_confidences; scale_confidences];
        cur_image_ids   = [cur_image_ids;   scale_image_ids];
        
    end
    
    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));
 
    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);
 
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
end
 
 
 
 

