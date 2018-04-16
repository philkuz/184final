
''' makes the L0 dataset '''
# coding: utf-8

# In[1]:


# %matplotlib inline


# In[2]:


from l0_gradient_minimization import l0_gradient_minimization_2d
import skimage.io as sio
import matplotlib.pyplot as plt
from demo_util import *
from glob import glob
import os
import time
pj = os.path.join


# In[3]:


# parameters
beta_max = 1.0e5
lmd = 0.05


# In[8]:



def prepare_data():
    val_names = []
    for folder in ['../fivek_dataset/rescale480p/']:
        for dirname in glob(pj(folder, '*')):
#             print('sup', dirname)
            for i in os.listdir(dirname):
#                 print(os.path.join(dirname, i))
                val_names.append(os.path.join(dirname, i))#test input image at 1080p
    return val_names


# In[9]:


val_names = prepare_data()


# In[10]:


out_dir = '../fivek_dataset/480p_L0_{lmd}/'.format(lmd=lmd)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# In[11]:



for val_file in val_names:
    if not os.path.isfile(val_file):
        continue
    contain_dir, filename = val_file.split('/')[-2:]
    #filename, ext = filename.split('.')
    #filename = '/'.join((contain_dir, filename)
    print(pj(out_dir, contain_dir, filename))
    if not os.path.exists(os.path.join(out_dir, contain_dir)):
        os.makedirs(os.path.join(out_dir, contain_dir))

    img = sio.imread(val_file)/ 255

#     scale = 480/max(read_image.shape) # * max(read_image.shape)
#     read_image = cv2.resize(read_image, (0,0), fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)

#     input_image=np.expand_dims(np.float32(read_image),axis=0)/255.0
#    print(max(input_image), min(input))
    st=time.time()
    output_image = l0_gradient_minimization_2d(img, lmd, beta_max=beta_max, beta_rate=2.0, max_iter=30, return_history=False)
    print("%.3f"%(time.time()-st))
    output_image=clip_img(output_image)
    sio.imsave(os.path.join(out_dir, contain_dir, filename) ,output_image)


# In[ ]:




