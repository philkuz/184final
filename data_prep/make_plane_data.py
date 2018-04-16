import pygame
import sys
import glob
from mesh_utils import *
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from numpy.matlib import repmat

#######################################################
#####         Select Plane Mesh Parameters        #####
#######################################################
num_points_width = 50
num_points_height = 50
width = 5
length = 7
theta = np.pi/6
phi = np.pi/5
disp = np.transpose(np.array([0, 0, -7.0, 1.0]))

#######################################################
#####           Make and Transform Meshes         #####
#######################################################

bare_mesh = make_plane_mesh(width,length,num_points_width,num_points_height)
bare_mesh = rotate_mesh(bare_mesh, theta, phi)
bare_mesh = translate(bare_mesh, disp)

texture_mesh = make_plane_mesh(width,length,2,2)
texture_mesh = rotate_mesh(texture_mesh, theta, phi)
texture_mesh = translate(texture_mesh, disp)

#pygame.init()
display = (800, 600)
screen = pygame.display.set_mode(
    display, pygame.DOUBLEBUF | pygame.OPENGL | pygame.OPENGLBLIT)

gluPerspective(45, display[0] / display[1], 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
draw_plane_mesh(bare_mesh,num_points_width,num_points_height,z_depth_shading)
pygame.image.save(screen, "theplane.png")
# for filename in glob.iglob('textures/*.png'):
#     #loadTexture(filename)
#     print(filename)
#     # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#     # draw_plane_mesh(bare_mesh,num_points_width,num_points_height,z_depth_shading)
#     # pygame.image.save(screen, "output/" + str(i) + "bare"+ ".png")
#     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#     draw_texture_plane(texture_mesh, filename)
#     file_id = str(filename[9:16])
#     print(file_id)
#     pygame.image.save(screen, "output/" + file_id+ ".png")
