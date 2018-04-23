<script type="text/javascript" async="" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML"></script>
# Applying Textures to Untextured Images Using Neural Networks

## Summary

We propose a method of applying textures automatically to non-textured images of 3d renderings using a fully-convolutional neural network. We take an untextured rendering, pass it into a trained Convolutional Neural Network that then outputs a (hopefully) proper textured result. This would remove the need to apply textures manually using mipmaps and texture coordinates, and potentially fixes issues like magnification of textures. The scope of this project focuses in particular on planes,
as a testbed for solutions that might work for more general geometries. 

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

## Experimental Results so far:

### Texture Cue 
Ultimately, we desired to pass in an image with a texture cue (ie painted by a user) that provides some information on what the texture should be inside of certain regions of the image. In the image below, you can see the net is able to learn the proper color of the output object - but it does not manage to copy the texture details of the cue. We suspect that this is the result of the L2 loss and explore other options in the next steps.
[![](https://d1b10bmlvqabco.cloudfront.net/attach/jcawl9n5m3s4s3/icguy9n240e1rp/jg490i8n2v22/all.jpg)](https://github.com/philkuz/184final/blob/master/images/geometry-v1.jpg)
### L0 Smoothing
As stated in the proposal, one of our dataset involved texturizing "texture-less" real images. We created such training pairs by using the L0 norm to wipe the high frequency textures from the image. In the result below, the left column shows the input, "texture-less" images, the middle represents the output of the network, and the right
is the ground-truth, textured images.
[![](https://d1b10bmlvqabco.cloudfront.net/attach/jcawl9n5m3s4s3/icguy9n240e1rp/jg49kul9iu17/individualImage.png)](https://github.com/philkuz/184final/blob/master/images/l0texturize.jpg)
Clearly the results are promising, but it makes sense that many textures are not preserved. Additionally, it appears that the network likes to hallucinate weird lines in the image as you can see in the sky pictures. This result is intriguing, but we realized it might be out of scope of the entire project, so we're including the result here, but
we will likely not continue down this route for the final project.
### Rotated Plane
Finally, we ran a test where we attempted to learn a geometric transformation, a proof of concept for another texture input method where we'd provide the texture directly, 
rather than through a texture cue. This method failed dramatically, as we had no extra parameters or degrees of freedom to ensure that a transform was learned. Upon reviewing the results, we realized the problem is a geometric transformation one that is already very difficult without an explicit geoemtric representation as part of the input, or at
very least a parameterization of the desired transform. AS we desire to learn an arbitrary, non-linear transform, we believe this approach will not be fruitful, and will primarily focus on the texture cue direction.
[![](images/rotated_plane_crop.jpg)](https://github.com/philkuz/184final/blob/master/images/rotated_plane.jpg)
## Successes and Next Steps
So far we've had difficulty getting these techniques to work. However a baseline test with style transfer, with the target texture suggests that we should be able to 
make some strides using a style loss as a part of our metric. 
<!-- TODO add style transfer results-->
We started training the network with  a style loss (using a separate VGG network, inline with [Johnson et al.](https://arxiv.org/abs/1603.08155) and here are some preliminary results.
[![](geometry-style-v2_partway_crop.jpg)](https://github.com/philkuz/184final/blob/master/images/geometry-style-v2_partway_crop.jpg)
The results are slightly hard to decode and its questionable whether this will help much. We're playing around with the weighting of this style loss with respect to the original objective, and the loss had not converged when this output was made. However, the style transfer result suggests this can be a very fruitful avenue of exploration.
### Generative Adversarial Networks
We are currently having issues with getting the network to produce nice, high frequency textures. We suspect that a part of the problem is our current MSE objective. We believe that learning an adversarial loss in line with [Pix2Pix](https://phillipi.github.io/pix2pix/). This is more of a stretch goal at the moment, but we're confident we'll be
able to take a stab at the problem using this model.

### Multiple orientations and scales
In all of our images you likely noticed that we're currenlty only testing a single sized plane in a single orientation. We are currently creating a dataset of multiple orientations and scales that we will train on after we get some success in better recreating the textures.

### Providing variable cues
For now, we've only provided texture cues in the center of the images. However, the dream would be to be able to provide cues in different parts of the image. If we have enough time, this will be a part of our final project as well.


### References:
[1] Chen, Qifeng, Jia Xu, and Vladlen Koltun. "Fast image processing with fully-convolutional networks." *IEEE International Conference on Computer Vision*. Vol. 9. 2017. [Link](http://openaccess.thecvf.com/content_iccv_2017/html/Chen_Fast_Image_Processing_ICCV_2017_paper.html)

[2] Fast Image Processing Code [Link](https://github.com/CQFIO/FastImageProcessing)

[3] MIT-Adobe FiveK Dataset [Link](https://data.csail.mit.edu/graphics/fivek/)

[4] Xu, Li, et al. "Image smoothing via L 0 gradient minimization." ACM Transactions on Graphics (TOG). Vol. 30. No. 6. ACM, 2011. [Link](https://dl.acm.org/citation.cfm?id=2024208)

[5] Nguyen, Rang MH, and Michael S. Brown. "Fast and effective L0 gradient minimization by region fusion." Proceedings of the IEEE International Conference on Computer Vision. 2015. [Link](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Nguyen_Fast_and_Effective_ICCV_2015_paper.pdf)

[6] Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "Perceptual losses for real-time style transfer and super-resolution." European Conference on Computer Vision. Springer, Cham, 2016.

[7] Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. arXiv preprint.
