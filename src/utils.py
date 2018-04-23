import os
import numpy as np
pj = os.path.join
def expand_dims_list(arr, axes):
    if type(axes) == int:
        axes = [axes]
    elif type(axes) != list:
        raise ValueError('axes must be list or int not {}'.format(type(axes)))
    for a in axes:
        arr = np.expand_dims(arr, axis=a)
    return arr
def make_texture_queue(no_texture_img, texture_img, area_mask):
    ''' grabs textured_img and rea_mask, then adds to no_texture_img as a "texture_queue" '''
    return no_texture_img * (1 - area_mask) + texture_img * area_mask

def circle_mask(mask_shape, center, radius):
    ''' makes a circular mask with radius and center '''
    a, b = center
    nx, ny  =mask_shape
    r = radius
    y, x = np.ogrid[-a:nx-a, -b:ny-b]
    mask = x*x + y*y <= r*r
    mask_array = np.zeros(mask_shape)
    mask_array[mask] = 1
    return mask_array

def get_center(shape):
    ''' returns a center of a rectangle '''
    return shape[0] / 2, shape[1] /2
