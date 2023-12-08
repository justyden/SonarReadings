load("resnet50_2_10_training.mat", "detector")
% connect to a webcam or ESP32-CAM
im = imread("TrainingData\ship\ship-275.png")                     % if reading a jpg file, use im = imread('image1.jpg') and remove loop
im = imresize( im, [224, 224] );
if size(im, 3) == 1
    im = ind2rgb(im, gray(256));
end
[bboxes, scores, labels] = detect(detector, im, 'Threshold', 0.12);    % apply the new network to the image
if ~isempty(bboxes)       % if at least one object was detected….
   im2 =  insertObjectAnnotation(im, 'rectangle', bboxes, cellstr(labels), 'Color', 'red');   
   imshow(im2)
    fprintf("The number of objects detected is %d\n", length(labels))
else
    imshow(im)
end


