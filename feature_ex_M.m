function fea=feature_ex_M(IMAGES)
%----------------------------Extract Feature------------------------------
if ndims(IMAGES)==3
    gray=rgb2gray(IMAGES);
    gray=imresize(gray,[50,50]);
else
    gray=IMAGES;
    gray=imresize(gray,[50,50]);
end
    glcm1 = graycomatrix(gray, 'offset',[0,3],'NumLevels', 9);
    glcm2 = graycomatrix(gray, 'offset',[1,3],'NumLevels', 9);
    glcm3 = graycomatrix(gray, 'offset',[2,3],'NumLevels', 9);
    glcm4 = graycomatrix(gray, 'offset',[3,3],'NumLevels', 9);
    glcm5 = graycomatrix(gray, 'offset',[3,2],'NumLevels', 9);
    glcm6 = graycomatrix(gray, 'offset',[3,1],'NumLevels', 9);
    glcm7 = graycomatrix(gray, 'offset',[3,0],'NumLevels', 9);
    glcm8 = graycomatrix(gray, 'offset',[3,-1],'NumLevels', 9);
    glcm9 = graycomatrix(gray, 'offset',[3,-2],'NumLevels', 9);
    glcm10 = graycomatrix(gray, 'offset',[3,-3],'NumLevels', 9);
    glcm11 = graycomatrix(gray, 'offset',[2,-3],'NumLevels', 9);
    glcm12 = graycomatrix(gray, 'offset',[1,-3],'NumLevels', 9);
    glcm13 = graycomatrix(gray, 'offset',[2,-2],'NumLevels', 9);
    glcm14 = graycomatrix(gray, 'offset',[2,0],'NumLevels', 9);
    fea=cat(3,glcm1,glcm2,glcm3,glcm4,glcm5,glcm6,glcm7,glcm8,glcm9,glcm10,glcm11,glcm12,glcm13,glcm14);
% end

% fea=cat(3,fea1,fea2);
fea=tensor(fea);
end
