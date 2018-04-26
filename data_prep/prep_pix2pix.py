''' preps the data for pix2pix, which takes in 256x256 images, where pairs are crammed into a single image'''
import numpy as np
from PIL import Image
import os
import sys
sys.path.append('../src/')
from utils import apply_texture_queue
pj = os.path.join

split = 'test'
version = 'v3'
data_dir = '../data_scratch/geometry-{}'.format(version)
out_basedir = pj(data_dir, 'pix2pix')

out_dir = pj(out_basedir, split)
if not os.path.exists(out_dir):
  os.makedirs(out_dir)

train_file = pj(data_dir, '{}-{}.txt'.format(split, version))
  # return image.resize((width * factor, height * factor))
def rescale(image, smallest_edge=256):
  width, height = image.size

  if height < width:
    factor = smallest_edge / height
  else:
    factor = smallest_edge / width
#   factor = int(factor)


  return image.resize((int(width * factor), int(height * factor)))

def crop_img(im, new_shape=(256, 256)):
  width, height = im.size   # Get dimensions
  new_width, new_height = new_shape

  left = (width - new_width)/2
  top = (height - new_height)/2
  right = (width + new_width)/2
  bottom = (height + new_height)/2

  return im.crop((left, top, right, bottom))
def prep_for_cue(im):
  return np.expand_dims(np.array(im), 0).astype(np.float32)/255.
def to_pillow(arr):
  return Image.fromarray(np.uint8(arr * 255.))
def make_cue(ipt, opt):
  ipt = prep_for_cue(ipt)
  opt = prep_for_cue(opt)
  cued = apply_texture_queue(ipt, opt, 'center_circle')
  return to_pillow(cued[0])
with open(train_file) as f:
  for line in f:
    ipt, opt = line.strip().split(',')
    # print(ipt, opt)
    input_image = Image.open(ipt)
    output_image = Image.open(opt)
    input_image= rescale(input_image)
    output_image= rescale(output_image)

    # crop the center of ipt and opt
    input_image = crop_img(input_image)
    output_image = crop_img(output_image)
    # for now add the cue as a preprocessing step
    input_image = make_cue(input_image, output_image)

    # combine ipt and opt
    new_image = Image.new('RGB', (512, 256))
    new_image.paste(output_image, (0, 0))
    new_image.paste(input_image, (256, 0))
    new_image.save(pj(out_dir, opt.split('/')[-1]))
    # input()



