%% An application that uses Alexnet to identify many objects.
picture = imread('TrainingData/plane/plane-001.png');
picture = (picture);
%nnet = alexnet; % Load the neural net
load("side_scan_network_2.mat", "side_scan_network_2")
picture = imresize(picture,[227, 227]); % Resize
label = classify(side_scan_network_2, picture); % Classify the picture
image(picture); % Show the picture
title(char(label)); % Show the label
drawnow;
