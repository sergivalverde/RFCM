% ************************************
% Robust Fuzzy-C-Means segmentation
%
% Sergi Valverde 
% ***********************************





%% 3D VOLUME (MRI SCAN)

clear all;

%% MRI scan
mri_brain = load_nifti('t1_brain');
MRI_brain = mri_brain.img;

options.info = 0;
options.gpu = 1;
options.beta = 1;
options.maxiter = 500;
tic;
[s,C, probability_maps] = rfcm(MRI_brain, 3, options);
toc;

probability_map3 = zeros(size(s));
probability_map3(:,:,:) = probability_maps(:,:,:,3);

mri_brain.img = s;
save_nifti(mri_brain,'segmentation_MRI');

mri_brain.img = probability_map3;
save_nifti(mri_brain,'probability_map');