
# coding: utf-8

# In[8]:


import rawpy
import imageio


# In[15]:


import os
pj = os.path.join


# In[32]:


fivek_dir = '../fivek_dataset/'
out_dir = pj(fivek_dir, 'pngs')
raw_dir = pj(fivek_dir, 'raw_photos')
in_dirs = []
out_dirs = []
conversion_tuples = []
for d in os.listdir(raw_dir):
    if os.path.isdir(pj(raw_dir, d)) and 'HQ' in d:
        in_dirs.append(pj(raw_dir, d, 'photos') )
        out_dirs.append(pj(out_dir,d))
        if not os.path.exists(out_dirs[-1]):
            os.makedirs(out_dirs[-1])
        for d in os.listdir(pj(raw_dir, d, 'photos')):
            img_name, ext = d.split('.')
            infile = pj(in_dirs[-1], d)
            outfile = pj(out_dirs[-1], '.'.join((img_name, 'png')))
            conversion_tuples.append((infile, outfile))




# In[39]:


def convert_pair(pair):
    in_file, out_file = pair
    with rawpy.imread(in_file) as raw:
        rgb = raw.postprocess()
        imageio.imsave(out_file, rgb)



import multiprocessing as mp
pool = mp.Pool(processes=12)
list(pool.imap_unordered(convert_pair, conversion_tuples))
