import pygame
import sys
import glob
import os
import pickle
from mesh_utils import *
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from numpy.matlib import repmat

#######################################################
#####          Set Up Texture Directories         #####
#######################################################

rot_texture_dir = '../rot_textures'
texture_dir = '../textures'
rot_planes_dir = '../rot_planes' 

if not os.path.exists(rot_texture_dir):
    os.makedirs(rot_texture_dir)
if not os.path.exists(rot_planes_dir):
    os.makedirs(rot_planes_dir)

#######################################################
#####         Select Plane Mesh Parameters        #####
#######################################################
num_points_width = 50
num_points_height = 50
width = 5
length = 5
num_rotations = 10
id_digit_len = math.ceil(math.log(num_rotations)/math.log(10))
len_texture_id = 7
disp = np.transpose(np.array([0, 0, -7.0, 1.0]))
min_z = -10
max_z = 0
shadin_fun = lambda point : z_depth_shading(point, min_z, max_z)

#######################################################
#####           Generate Random Rotations         #####
#######################################################

rotations = [sample_sphere() for _ in range(num_rotations)]
with open(rot_planes_dir + '/rotation_list.pkl', 'wb') as f:
	pickle.dump(rotations, f)

#######################################################
#####           Make and Transform Meshes         #####
#######################################################

texture_base_mesh = make_plane_mesh(width,length,2,2)

bare_mesh = make_plane_mesh(width,length,num_points_width,num_points_height)
rotated_base_meshes = []

pygame.init()
display = (800, 600)
screen = pygame.display.set_mode(
    display, pygame.DOUBLEBUF | pygame.OPENGL | pygame.OPENGLBLIT)

gluPerspective(45, display[0] / display[1], 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

for rot_ind in range(len(rotations)):
	rotation = rotations[rot_ind]
	rot_mesh = rotate_mesh(bare_mesh, rotation[0], rotation[1])
	rot_mesh = translate(rot_mesh, disp)
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
	draw_plane_mesh(rot_mesh,num_points_width,num_points_height,shadin_fun)
	pygame.image.save(screen, rot_planes_dir + '/' + get_digit_id(id_digit_len, rot_ind) + '.png')
	#make the base mesh the textures will go on
	texture_mesh = rotate_mesh(texture_base_mesh, rotation[0], rotation[1])
	texture_mesh = translate(texture_mesh, disp)
	rotated_base_meshes.append(texture_mesh)

len_tex_dir = len(texture_dir)
for filename in glob.iglob(texture_dir + '/*.png'):
	for rot_ind in range(len(rotated_base_meshes)):
		base_mesh = rotated_base_meshes[rot_ind]
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		draw_texture_plane(base_mesh, filename)
		file_id = str(filename[len_tex_dir:len_tex_dir+len_texture_id+1])
		file_name = file_id + '_' + get_digit_id(id_digit_len, rot_ind)
		pygame.image.save(screen, rot_texture_dir + '/' + file_name + ".png")
