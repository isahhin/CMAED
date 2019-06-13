%========================================================================================================================
% Matlab code for CMAED 2019 paper
% Copyright: Sahin ISIK, 2019
%
% link: https://github.com/isahhin/cmaed
% It is restricted to use for personal and scientific research purpose only
% No Warranty
%       (1) "As-Is". Unless otherwise listed in this agreement, this SOFTWARE PRODUCT is provided "as is," with all faults, defects, bugs, and errors.
%       (2 )No Warranty. Unless otherwise listed in this agreement.
% Please cite the following paper when used this code:
%   1. Işık, Şahin, and Kemal Özkan. "Common matrix approach-based multispectral image fusion and its application to edge detection." 
%      Journal of Applied Remote Sensing 13, no. 1 (2019): 016515.
%========================================================================================================================


clear all
clc 
close all


name = 'hyperspectral_dataset'

load (['database\', name, '\PaviaU.mat'] )
load (['database\', name, '\PaviaU_gt.mat'] )
gt = paviaU_gt;
gt = edge(gt, 'Canny');
mxn=size(gt);
%gt=imresize(gt,[512,256]);
figure;imshow(abs(gt),[])
nx=1;
mx=1;
c=paviaU;
sigma=1;
if 1
   
    for ii=nx:size(c,3)-mx
       
        d=c(:,:,ii);
        d=d-min(d(:));
        d=d./max(d(:));
        
        dataSet(:,:,ii)=d;
        [ii max(d(:)) min(d(:))]
        [Gx, Gy] = smoothGradient(dataSet(:,:,ii),sigma);
        rg=hypot(Gx,Gy);
               
        resMag(:,:,ii)=single(rg)./max(rg(:));
    end
    Cmag=reshape(resMag,size(resMag,1)*size(resMag,2),size(resMag,3));
    meanref=mean(Cmag,2);
    refmag=Cmag(:,end)-meanref;
    for i=1:size(Cmag,2)-1
        B(:,i)=(Cmag(:,i)-meanref)-refmag(:);
    end
    rank(Cmag)
    [u,s]=qr(double(B),0);
    diffMag=0*refmag;
    for ii=1:size(B,2)
        diffMag=diffMag+dot(u(:,ii),refmag(:))*refmag(:);
    end
    comMag=refmag-(diffMag)+meanref;
    Cmag=reshape(comMag,size(resMag,1),size(resMag,2));
    Dmag=reshape(diffMag,size(resMag,1),size(resMag,2));
    refmag=reshape(refmag,size(resMag,1),size(resMag,2));

    save Cmag Cmag
    save Dmag Dmag
    save refmag refmag
    save Gx Gx
    save Gy Gy
else
    load Cmag.mat   
    load Dmag Dmag
    load refmag refmag
    load Gx Gx
    load Gy Gy
end
    Cmag=imresize(Cmag,mxn); 

    [edgeCom,thresh] = my_edge(abs(Cmag),Gx,Gy,0.92,0.1);
    h=figure;imshow(~edgeCom) 
    iptsetpref('ImshowBorder','tight');
    hgexport(h, [ name, '_CMAED'], hgexport('factorystyle'), 'Format', 'bmp', 'Resolution', 320);

