'''
make the train_test_split for the original images data
'''
import os
from glob import glob
from shutil import copyfile
from random import shuffle
pj = os.path.join
abspath = os.path.abspath
fivek_root = 'fivek_dataset'
# TODO update this to reflect new dir structure
data_root = 'data'
textured_out = pj(data_root, 'texture')
untextured_out = pj(data_root, 'no_texture')
split = 0.8

if not os.path.exists(data_root):
    os.makedirs(data_root)
if not os.path.exists(textured_out):
    os.makedirs(textured_out)
if not os.path.exists(untextured_out):
    os.makedirs(untextured_out)
textured_paths  = pj(fivek_root,'rescale480p')
untextured_paths = pj(fivek_root, '480p_L0_0.05')
img_idx = 0
mapping = []
copy_file = False
for dirname in os.listdir(textured_paths):
    tex_path = pj(textured_paths, dirname)
    untex_path = pj(untextured_paths, dirname)
    for img in os.listdir(tex_path):
        if 'png' not in img:
            continue
        base_file = '{0:07d}.png'.format(img_idx)
        tex_out = pj(textured_out, base_file)
        untex_out = pj(untextured_out, base_file)
        if not os.path.exists(tex_out):
            copyfile(pj(tex_path, img), tex_out)
            copy_file  = True
        if not os.path.exists(untex_out):
            copyfile(pj(untex_path, img), untex_out)
            copy_file = True
        mapping.append((pj(dirname, img), base_file))
        img_idx+=1
if copy_file:
    with open(pj(data_root, 'mapping.txt'), 'w') as f:
        for line in mapping:
            f.write(','.join(line) + '\n')

# create train/test files
file_tuples = []
for filepath in os.listdir(textured_out):
    input_path = abspath(pj(untextured_out, filepath))
    output_path = abspath(pj(textured_out, filepath))
    file_tuples.append((input_path, output_path))

shuffle(file_tuples)
split_idx = int(len(file_tuples) * split)
train_list = file_tuples[:split_idx]
test_list = file_tuples[split_idx:]
with open(pj(data_root, 'train.txt'), 'w') as f:
    for line in train_list:
        f.write(','.join(line) + '\n')
with open(pj(data_root, 'test.txt'), 'w') as f:
    for line in test_list:
        f.write(','.join(line) + '\n')




