im = imread('Test_Data\IMG_2518.JPEG'); % load in a test image
im = imresize( im, [224, 224] ); % resize; NOTE: YOLOv2 also works for images larger than [224 224]
load("detector.mat", "detector")
[bboxes, scores, labels] = detect(detector, im, 'Threshold', 0.5); % apply the new trained network to the image
if ~isempty(bboxes) % if at least one object was detected....
    im2 = insertObjectAnnotation(im, 'rectangle', bboxes, cellstr(labels), 'Color', 'red'); % draw red box
    imshow(im2)
    fprintf("The number of objects detected is %d\n", length(labels)) % for testing purposes
    for i = 1 : size(bboxes, 1) % for testing purposes; comment out for better performance
        fprintf("Object # %d", i) % find centroid of bounding box
        centroidX = bboxes(i,1) + bboxes(i,3)/2 % x is the column value here
        centroidY = bboxes(i,2) + bboxes(i,4)/2 % y is the row value here
    end
else
    imshow(im)
end