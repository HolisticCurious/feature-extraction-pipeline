# feature-extraction-pipeline
A pipeline for the calculation of morphological and textural features, in python, from images. 

The notebook includes the steps of importing images, calculating morphological and textural features. Can be turned into .py file.
 

*   Initial images may be in any format, but the code may need changes. Currently, it is using cv's imread (see [documentation](https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56)).
*   Images as processed as grayscale. If multi-channel, channels will be merged according to current code. If the analysis is intended to look at individual channels, split image into different files.
*   The code also includes to option to import a label, in the format csv (with image name and numerical label). This is, of course, not a mandatory step; feature extraction can be performed without classification.
*   GLCM features cannot be calculated in float images. As such, intensity values may have to be normalized prior to feature calculation.
*   Sample images in sample_imgs folder (segmented nuclei).
*   Code prepared to be run on Google Colaboratory (upload feature calculation .py files) but can be run locally.

Still working on Data Visualization.

I thank Kirill Lavrenyuk @Dahl Lab for Subcellular Structure and Engineering, Carnegie Mellon University, for the example image.
