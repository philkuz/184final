'''
make the train_test_split for the original images data
'''
import os
from glob import glob
from shutil import copyfile
from random import shuffle
pj = os.path.join
abspath = os.path.abspath
### PARAMS ###
src_data_root = '../data/geometry-v1'
textured_out = pj(src_data_root, 'texture')
split = 0.8
version = 'v2'
### END PARAMS ###
if version == 'v1':
    one_path =  True
    input_dir = pj(src_data_root, 'no_texture')
else:
    one_path = False
    input_dir = pj(src_data_root, 'src_textures')

# create train/test files
file_tuples = []
for filepath in os.listdir(textured_out):
    if "png" not in filepath:
        continue
    if one_path:
        input_path = abspath(pj(input_dir, '{0:07d}.png'.format(0)))
    else:
        input_path = abspath(pj(input_dir, filepath))
    output_path = abspath(pj(textured_out, filepath))
    file_tuples.append((input_path, output_path))
    if not (os.path.exists(file_tuples[-1][0]) and os.path.exists(file_tuples[-1][1])):
        print(file_tuples[-1], "don't exist")

shuffle(file_tuples)
split_idx = int(len(file_tuples) * split)
train_list = file_tuples[:split_idx]
test_list = file_tuples[split_idx:]
out_root = 'data/geometry-{}'.format(version)
if not os.path.exists(out_root):
    os.makedirs(out_root)
with open(pj(out_root, 'train-{}.txt'.format(version)), 'w') as f:
    for line in train_list:
        f.write(','.join(line) + '\n')
with open(pj(out_root, 'test-{}.txt'.format(version)), 'w') as f:
    for line in test_list:
        f.write(','.join(line) + '\n')




