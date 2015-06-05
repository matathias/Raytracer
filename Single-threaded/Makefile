###############################################################################
# CS/CNS 171 Fall 2014
#
# This is a template Makefile for OpenGL programs. Edit it however you find
# convenient.
# 
# The current version of this file should compile OpenGL programs just fine on
# Debian-based Linux operating systems.
#
# If you run Mac OS or other distributions of Linux, then you may have to
# fool around with the 'INCLUDE' and 'LIBDIR' lines below before the Makefile
# can compile the OpenGL parts successfully.
###############################################################################
CC = g++
FLAGS = -g -o

# Change the fourth entry in this line to point to the Eigen directory
INCLUDE = -I/usr/X11R6/include -I/usr/include/GL -I/usr/include -I/home/warrick/Dropbox/Raytracer/Single-threaded

LIBDIR = -L/usr/X11R6/lib -L/usr/local/lib
SOURCES = util.cpp raytrace.cpp
LIBS = -lGLEW -lGL -lGLU -lglut -lm

EXENAME = raytrace

all: raytrace

raytrace: $(SOURCES)
	$(CC) $(FLAGS) $(EXENAME) $(INCLUDE) $(LIBDIR) $(SOURCES) $(LIBS)

clean:
	rm -f *.o $(EXENAME)

.PHONY: all clean

