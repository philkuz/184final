import pygame
import sys
import glob
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

vertices = (
    # x  y  z
    (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, -1.0),
)

edges = (
    (0, 1),
    (0, 3),
    (2, 1),
    (2, 3),
)


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


def draw_cube(lines=False):
    if lines:
        glBegin(GL_LINES)
        for edge in edges:
            glColor3fv((1, 1, 1))
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()
    else:
        glBegin(GL_QUADS)
        test_coord = np.array

        glTexCoord2f(0.0, 0.0)
        glVertex3f(-1.0, -1.0, 5.0)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(1.0, -1.0, 5.0)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(1.0, 1.0, 1.0)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(-1.0, 1.0, 1.0)



        # glTexCoord2f(1.0, 0.0)
        # glVertex3f(-1.0, -1.0, -1.0)
        # glTexCoord2f(1.0, 1.0)
        # glVertex3f(-1.0, 1.0, -1.0)
        # glTexCoord2f(0.0, 1.0)
        # glVertex3f(1.0, 1.0, -1.0)
        # glTexCoord2f(0.0, 0.0)
        # glVertex3f(1.0, -1.0, -1.0)
        # glTexCoord2f(0.0, 1.0)
        # glVertex3f(-1.0, 1.0, -1.0)
        # glTexCoord2f(0.0, 0.0)
        # glVertex3f(-1.0, 1.0, 1.0)
        # glTexCoord2f(1.0, 0.0)
        # glVertex3f(1.0, 1.0, 1.0)
        # glTexCoord2f(1.0, 1.0)
        # glVertex3f(1.0, 1.0, -1.0)
        # glTexCoord2f(1.0, 1.0)
        # glVertex3f(-1.0, -1.0, -1.0)
        # glTexCoord2f(0.0, 1.0)
        # glVertex3f(1.0, -1.0, -1.0)
        # glTexCoord2f(0.0, 0.0)
        # glVertex3f(1.0, -1.0, 1.0)
        # glTexCoord2f(1.0, 0.0)
        # glVertex3f(-1.0, -1.0, 1.0)
        # glTexCoord2f(1.0, 0.0)
        # glVertex3f(1.0, -1.0, -1.0)
        # glTexCoord2f(1.0, 1.0)
        # glVertex3f(1.0, 1.0, -1.0)
        # glTexCoord2f(0.0, 1.0)
        # glVertex3f(1.0, 1.0, 1.0)
        # glTexCoord2f(0.0, 0.0)
        # glVertex3f(1.0, -1.0, 1.0)
        # glTexCoord2f(0.0, 0.0)
        # glVertex3f(-1.0, -1.0, -1.0)
        # glTexCoord2f(1.0, 0.0)
        # glVertex3f(-1.0, -1.0, 1.0)
        # glTexCoord2f(1.0, 1.0)
        # glVertex3f(-1.0, 1.0, 1.0)
        # glTexCoord2f(0.0, 1.0)
        # glVertex3f(-1.0, 1.0, -1.0)
        glEnd()

#pygame.init()
display = (800, 600)
screen = pygame.display.set_mode(
    display, pygame.DOUBLEBUF | pygame.OPENGL | pygame.OPENGLBLIT)


i = 0
for filename in glob.iglob('textures/*.png'):
    loadTexture(filename)

    gluPerspective(45, display[0] / display[1], 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    draw_cube(lines=False)
    pygame.image.save(screen, "output/" + str(i) +  ".png")
    i += 1
