function NMI_v=NMI(y,y_r)
M=max(y_r);
N=size(y(:),1);
Iy=0;
Iy_r=0;
MI=0;
for i=1:M
    for j=1:M
        py=size(find(y(:)==i),1)/N;
        py_r=size(find(y_r(:)==j),1)/N;
        py_y_r=size(find(y(:)==i&y_r(:)==j),1)/N;
        Iy=Iy-py*log2(py);
        Iy_r=Iy_r-py_r*log2(py_r);
        if py_y_r~=0
            MI=MI+py_y_r*log2(py_y_r/(py*py_r));
        end
    end
end
NMI_v=MI/(max([Iy,Iy_r])/M);
end