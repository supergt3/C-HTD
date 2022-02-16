function fea=tensor_Construction(dirStrings)
imgList = dir(dirStrings);
dirS=dirStrings(1:length(dirStrings));
IMAGES = cell(1, length(imgList)-2);
for i = 1 : length(imgList)-2
    IMAGES{i} = im2double(imread([dirS imgList(i+2).name]));
%     %% Resize to make memory efficient
%     if max(size(IMAGES{i})) > 1000 || length(imgList) > 10,
%         IMAGES{i} = imresize(IMAGES{i}, [80,80]);
%     end
end
disp('Images loaded. Beginning feature detection...');
fea=cell(1,length(imgList)-2);
for i = 1 : length(imgList)-2
    fea{i}=feature_ex_M(IMAGES{i});
    i
end
end
