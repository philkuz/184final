import os
import glob

file_num = 0
data_dir = 'textures'
for filename in glob.iglob(data_dir+'/*.jpg'):
	file_num_str = str(file_num)
	new_filename = data_dir + '/' + (7-len(file_num_str))*'0' + file_num_str +'.png'
	os.rename(filename, new_filename)
	file_num += 1