# Robust Fuzzy C-Means clustering for MRI images

Robust Fuzzy C-Means clustering for MRI images as described in [Dzung L. Pham 2000](http://www.sciencedirect.com/science/article/pii/S1077314201909518). The method is useful for rapid prototyping when a fast tissue segmentation is needed. The method has been implemented with GPU capabilities. 

## How to use the method:

```
mri_brain = load_nifti('t1_brain');
MRI_brain = mri_brain.img;

options.info = 0;
options.gpu = 1;
options.beta = 1;
options.maxiter = 500;
classes = 3;

[s] = rfcm(MRI_brain, classes, options);

mri_brain.img = s;
save_nifti(mri_brain,'segmentation_MRI');

```

## Available options:

In general, all the available options in the software are set to values that are known to work well in most of the cases. However, each of these can be tuned in the `options` variable passed as an input:


+`options.weighting`: Fuzzy factor exponent in Fuzzy C-means clustering (default 2)

+`options.maxiter`: Number of maximum iterations during energy minimization FCM (default 200)

+`options.num_neigh`: Radius of the neighborhood used in spatial contraint (default 1)

+`options.dim`: Dimension of the neighborhood (default 2)

+`options.term`:  Minimum error in energy minimization to stop (default 1E-3)

+`options.gpu`: Use GPU (default 0)

+`options.info`:  Show information during tissue segmentation (default 0)


## Notes:

+ Image sections that have not to be segmented should be masked as zero in the input image for better performance.
+ The previous example uses the nifti_toolbox available [here](https://github.com/sergivalverde/nifti_tools).


# Credits:

[NeuroImage Computing Group](http://atc.udg.edu/nic/research.html), VICOROB [Vision and Robotics Institute](vicorob.udg.edu), University of Girona.

