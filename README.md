<script type="text/javascript" async="" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML"></script>
# Applying Textures to Untextured Images Using Neural Networks

## Summary

We propose a method of applying textures automatically to non-textured images of 3d renderings using a fully-convolutional neural network. We take an untextured rendering, pass it into a trained Convolutional Neural Network that then outputs a (hopefully) proper textured result. This would remove the need to apply textures manually using mipmaps and texture coordinates, potentially fixes issues like magnification of textures, and remove. This project does not automatically create textures for application on the 3D object. Instead, it acts like the last stage of a graphics pipeline, but if it works can quickly apply realistic looking texture. We train our network on a similar task - rendering textures for “untextured images” generated by applying an $$L_0$$ filter to smooth out 
the non-important edges in 

## Problem Description

Producing textures for images is a labor intensive process that can take significant human time. Additionally, older source material, such as video games from antiquated hardware, lacks textural detail of the level modern systems contain. Finding a fast and effective way to automatically apply textures to untextured images could significantly reduce labor for projects that require handmade textures. Automatic texture generation is challenging due to the level of detail and consistency with the scene needed to produce believable and pleasing results.  We propose a method, as a final step in a graphics pipeline, for utilizing neural nets to generate textures on non-textured images. 

|![Source Image (“Untextured Rendering”, $$L_0$$ of Ground Truth)](https://d2mxuefqeaa7sj.cloudfront.net/s_8DD81CC4A167A6BC0747207D4F08D74E4063E97814787C9C3CF8C3FC912A5AC4_1522619922117_L0.jpeg)|
|:--:| 
| *Source Image (“Untextured Rendering”, $$L_0$$ of Ground Truth)* |
|![Target Image (“Texture Image”, Ground Truth)](https://d2mxuefqeaa7sj.cloudfront.net/s_8DD81CC4A167A6BC0747207D4F08D74E4063E97814787C9C3CF8C3FC912A5AC4_1522619981475_Reg.jpeg)|
|:--:|
| *Target Image (“Texture Image”, Ground Truth)* |


To train the neural network, we need a dataset that contains texture-less images and ground-truth textured images. Generating this data from existing 3d models would be time-intensive and cost-prohibitive. However, we can make a pseudo-dataset by applying $$L_0$$ gradient minimization to real photographs and train the network to output the textured image. 

|![Normal Graphics Pipeline to our proposed graphics pipeline](https://d2mxuefqeaa7sj.cloudfront.net/s_EC8A632A5A7918595C67F233EC67BB235C13597A531880ECA5FA4896F0896040_1522622416920_diag.png)|
|:--:|
| *Normal Graphics Pipeline to our proposed graphics pipeline* |



At test time, we will then feed textureless images to the trained neural net in a graphics pipeline which will then apply approximate textures to the image.  Ideally, this should be able to take in a rendered image like the one below and output a nice texture. It’s likely that to get a good result, we’d have to apply $$L_0$$ to the rendering to get the result.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_EC8A632A5A7918595C67F233EC67BB235C13597A531880ECA5FA4896F0896040_1522623965829_image.png)

## Goals and Deliverables
### Quality Metrics

Quantitative Image Processing Metrics

- SSIM (Structural similarity) calculated with output images and ground truth images 
- PSNR (Peak signal-to-noise ratio) calculated with output images and ground truth images 

Qualitative Metrics

- We will produce a google survey where a set of randomly selected images from our dataset will be stripped of their texture and have texture reapplied with our method. We will then ask respondents to rank the subjective quality of the images on a 5-point scale. 
- If we reach our secondary goal, the same processes will be pursued with a randomly selected set of curated images from older video games and images rendered without textures.

### Baseline Deliverables

Acceptably reproduce textures on real images which have had their textures removed. Acceptability will be quantified by a combination of standard quantifiable metrics in image processing, SSIM and PSNR, and qualitative metrics, classmate submissions to a survey on the quality of images produced. 

### Ideal Deliverables

Acceptably apply textures to poorly textured images derived from old video games or renderings made without textures. As an exact ground-truth will not be present for these images, quality will be judged on qualitative metrics alone. 

We’d also like to try a method of “painting” a texture onto the images as well, providing contextual clues what kind of texture should be in the rest of the image. This would require a slight change to the data generation code that might be possible to do if we get ahead of schedule.

### Questions
- Can believable textures be applied by neural nets to textureless images?
- Can rendered images with poor textures be believably enhanced with neural nets? 

## Schedule

**April 2-8**

- Produce Processing structure for dataset
  - Produce framework to apply $$L_0$$ gradient-minimization to images
  - Integrate base Fast Image Processing code [2] into a pipeline for the given image task

**April 9-15**

- Build data set
  - Access MIT-Adobe FiveK image set
  - Create set of $$L_0$$ gradient-minimized images from MIT-Adobe FiveK image set
- Familiarize with library

**April 16-22**

- Train neural networks on dataset
  - Build content aggregation network
  - Train network on $$L_0$$gradient-minimized images as 
- Milestone Deliverables
  - Status Report Webpage
    - Reflect on results and progress
  - Video
    - Summarize progress with sample outputs
    - Code screenshots
  - Presentation Slides
    - Summarize progress so far

**April 23-May 1**

- Hyperparameter Tuning
  - Tinker with network parameters
  - Create best possible images as defined by aforementioned parameters
- Poster Presentation
  - Refine slides to present
- Final Website
  - Abstract of project
  - Final results
  
## Resources
- Computational Resources
  - 2 Nvidia 1080s on a home server
- Datasets
  - MIT-Adobe FiveK Dataset [Link](https://data.csail.mit.edu/graphics/fivek/)
- Software Resources
  -  Fast Image Processing Code [Link](https://github.com/CQFIO/FastImageProcessing)

### References:
[1] Chen, Qifeng, Jia Xu, and Vladlen Koltun. "Fast image processing with fully-convolutional networks." *IEEE International Conference on Computer Vision*. Vol. 9. 2017. [Link](http://openaccess.thecvf.com/content_iccv_2017/html/Chen_Fast_Image_Processing_ICCV_2017_paper.html)

[2] Fast Image Processing Code [Link](https://github.com/CQFIO/FastImageProcessing)

[3] MIT-Adobe FiveK Dataset [Link](https://data.csail.mit.edu/graphics/fivek/)

