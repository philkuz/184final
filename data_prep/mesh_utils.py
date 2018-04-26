import pygame
import sys
import glob
import random
import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from numpy.matlib import repmat

def loadTexture(textureFileName):
    textureSurface = pygame.image.load(textureFileName)
    textureData = pygame.image.tostring(textureSurface, "RGBA", 1)
    width = textureSurface.get_width()
    height = textureSurface.get_height()

    glEnable(GL_TEXTURE_2D)
    texid = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D, texid)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData)

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    return texid


def translate(points, disp):
    '''
        Translates every point in points by disp
    '''
    trans_mat = np.identity(4)
    trans_mat[:,3] = disp
    return np.dot(trans_mat, points)

def rotate_mesh(mesh, theta, phi):
    '''
        Mathematical spherical coords, phi rotation in z-x plane (about y), theta rotation in x-y plane (about z) 
    '''
    rot_about_y_mat = np.array([
            [ np.cos(phi), 0,  np.sin(phi), 0],
            [0,            1,            0, 0],
            [-np.sin(phi), 0,  np.cos(phi), 0],
            [0,            0,            0, 1]     
        ])
    rot_about_z_mat = np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta),  np.cos(theta), 0, 0],
            [0,                           0, 1, 0],
            [0,                           0, 0, 1]
        ])
    return np.dot(rot_about_z_mat, np.dot(rot_about_y_mat, mesh))

def make_plane_mesh(width, length, num_points_width, num_points_height): 
    mesh = np.zeros((4, num_points_width*num_points_height))
    x_sep = width/(num_points_width-1)
    y_sep = length/(num_points_width-1)
    #build the mesh with the bottom corner on the origin
    for y in range(num_points_height):
        for x in range(num_points_width):
            mesh[:, y*num_points_width + x] = np.array([x_sep*x, y_sep*y, 0.0, 1.0], dtype=np.float32)
    #shift the mesh so the middle is at the origin
    displacement = np.transpose(np.array([-width/2, -length/2, 0.0, 1.0]));
    displacement.reshape((4,1))
    mesh = translate(mesh, displacement)
    return mesh

def z_depth_shading(point, z_min=-10, z_max=2):
    z_value = point[2]
    r_val = (z_value - z_min)/ (z_max - z_min)
    if r_val < 0:
        r_val = 0.0
    if r_val > 1.0:
        r_val = 1.0
    return (r_val, 0.0, 0.0)

def draw_texture_plane(mesh, texture_filename):
    loadTexture(texture_filename)
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(*mesh[:3,0])
    glTexCoord2f(1.0, 0.0)
    glVertex3f(*mesh[:3,1])
    glTexCoord2f(1.0, 1.0)
    glVertex3f(*mesh[:3,3])
    glTexCoord2f(0.0, 1.0)
    glVertex3f(*mesh[:3,2])
    glEnd()

def sample_sphere():
    u = random.uniform(0, 1)
    v = random.uniform(0, 1)
    theta = 2*math.pi*u 
    phi = math.acos(2*v - 1)
    #if theta or phi > pi, reflect back to (0, pi)
    if theta > math.pi:
        offset_theta =  theta - math.pi
        theta = theta - 2*offset_theta
    if phi > math.pi:
        offset_phi = phi - math.pi
        phi = phi - 2*offset_phi
    #center theta and phi from (0, pi) to (-pi/2, pi/2)
    theta = theta - math.pi/2
    phi = phi - math.pi/2
    return (theta, phi)

def get_digit_id(digits, idx):
    idx_str = str(idx)
    if len(idx_str) > digits:
        raise ValueError('Index out of bounds for ' + str(digits) + ' digit numbers')
    return (digits-len(idx_str))*'0' + idx_str



def draw_plane_mesh(mesh, num_points_width, num_points_height, shading_fun):
    glBegin(GL_QUADS)
    for y_corner_ind in range(num_points_height-1):
        for x_corner_ind in range(num_points_width-1):
            p0 = mesh[:3, x_corner_ind + num_points_width*y_corner_ind]
            glColor3fv(shading_fun(p0))
            glVertex3f(*p0)
            p1 = mesh[:3, x_corner_ind + 1 + num_points_width*y_corner_ind]
            glColor3fv(shading_fun(p0))
            glVertex3f(*p1)
            p2 = mesh[:3, x_corner_ind + 1 + num_points_width*(y_corner_ind + 1)]
            glColor3fv(shading_fun(p0))
            glVertex3f(*p2)
            p3 = mesh[:3, x_corner_ind + num_points_width*(y_corner_ind + 1)]
            glColor3fv(shading_fun(p0))
            glVertex3f(*p3)
            glColor3fv([1.0,1.0,1.0])
    glEnd()
