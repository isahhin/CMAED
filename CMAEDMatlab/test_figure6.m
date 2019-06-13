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


clear 
clc 
close all

% fname = 'fake_and_real_food';
% fname = 'egyptian_statue';
fname = 'real_and_fake_apples'
name = 'complete_ms_data'

if 0

    oriFiles = dir(['database\',name,'\*.']);  % the folder in which our images exists
    filenameOI = strcat('database\',name,'\',fname,'_ms\', fname,'_ms');
    gt=imread(strcat('database\',name,'\',fname,'_ms\',fname,'_ms\',fname,'_RGB.bmp'));
    
    %figure; imshow((gt))
    %gt1 = rgb2gray(gt); 
    %gt = edge(gt1,'canny');


    imgsFilesPNG = dir(strcat(filenameOI,'\*.png'));
    imagePNG = strcat(filenameOI,'\',imgsFilesPNG(1).name);
    imgData = im2single(imread(imagePNG));
    imgData = imgData(:,10:end,:);
    
    %figure;imshow(imgData,[])
    noOfSamples = length(imgsFilesPNG);
    [h, w] = size(imgData);
    dataSet = zeros(h, w, noOfSamples);
    magnitudes = zeros(h, w,noOfSamples);
    Cmag = zeros(h*w,noOfSamples);
    %compute magnitudes of images
    sigma=2; %image smoothing parameter
    for ii=1:length(imgsFilesPNG)
         imagePNG = strcat(filenameOI, '\',imgsFilesPNG(ii).name);
         imgData = single(imread(imagePNG));
         imgData = imgData(:,10:end,:)./max(imgData(:));  %normalizing  
         dataSet(:,:,ii) = imgData;
         [Gx, Gy] = smoothGradient(imgData,sigma);
         mag = hypot(Gx,Gy);
         %figure; imhsow(Gx, [])
         %figure; imhsow(Gy, [])
         %figure; imhsow(mag, [])
         magnitudes(:,:,ii) = single(mag);
         Cmag(:, ii) = mag(:);
    end

    meanref = mean(Cmag,2); %meanref : mean of magnitude
    referenceMag = Cmag(:,noOfSamples); % takes the last one as reference
    
     %refmag : reference magnitude image after mean removal
    refmag = referenceMag - meanref;
    B = zeros(h*w, noOfSamples-1); % B: difference subspace
    for i=1:noOfSamples-1
        B(:,i) = Cmag(:,i) - meanref - refmag;
    end

    % gram schmidt orthogonalization on difference subspace (B)
    [u, s] = qr(B,0);
    
    %in python
    %u, s = np.linalg.qr(a, mode='economic')
    
    % difference vector assoicated with reference magnitude (refmag)
    diffMag=0*refmag; 
    
    for ii=1:noOfSamples-1
        diffMag = diffMag + dot(u(:,ii), refmag)*refmag;
    end
    
    % common magnitude assoicated with reference magnitude (refmag)
    comMag = refmag - diffMag + meanref;  
    Cmag = reshape(comMag, h, w);
    Dmag = reshape(diffMag, h, w);
    refmag = reshape(refmag, h, w);
    
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


f1=figure;imshow(Cmag,[]);  title('Common Magnitude used for edge detection');
f2=figure;imshow(Dmag,[]);  title('Difference Magnitude');
f3=figure;imshow(refmag,[]); title('Reference Magnitude');



 [edgeCom,thresh] = my_edge(abs(Cmag), Gx, Gy,0.80,0.2);
 h=figure;imshow(~edgeCom) 
% iptsetpref('ImshowBorder','tight');
% hgexport(h, ['results/Figure6/', fname, '_CMAED'], hgexport('factorystyle'), 'Format', 'bmp', 'Resolution', 320);

