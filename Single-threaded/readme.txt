David Warrick
CS/CNS 171 Fall 2014
Assignment 6

The main program, raytrace, is located within the same directory as this readme 
file. Open the make file and change the last include directory to correctly 
point to Eigen; then, simply run "make" from within the directory to compile the
program.

util.cpp and util.h are helper files that contain useful code used in the main 
program such as superquadric functions and functions to turn a 4x4 matrix into a
3x3 matrix and vice versa.


/------------------------------------------------------------------------------\
|          ~~~ raytrace ~~~                                                    |
\------------------------------------------------------------------------------/

To execute the program, type
    ./raytrace [extra arguments]
where [extra arguments] can be any of several input arguments.
Note: instead of inputting arguments from the commandline, a single .txt file
      can be specified. The file must contain the same arguments as with command
      line entry, though the obvious advantage is that typing a single file name
      takes much less time and effort than entering the required arguments for
      multiple objects - not to mention making edits to a large entry is easy, 
      too.
      The only required delimiter in the .txt file is a space. Commands do not
      need to be separated by a newline, though doing so makes the file look
      cleaner. Also, commands do not need to be input in any specific order.
      Note, however, that the first entered material will be associated with the
      first entered object. If you desire to define the material of the nth
      object, then you will need to manually define the materials of all n-1
      objects as well. If no material is specified, then the object defaults to
      a material with (.5, .5, .5) diffuse, (1, 1, 1) specular, 10 shininess,
      a snell ration of .9, and an opacity of .01.
      Additionally, because of how the file and argument parsers work, any text
      that is not part of an input command or its arguments is ignored.

If no input file is specified, then the produced image will be output in a file
of the name "out.ppm". If an input file /is/ specified, then the produced image
will be output in a file of the same name as the input file, but with the
file extension ".ppm" instead of ".txt".

All output files are put in the "output" directory.

Inputs:
[-res Nx Ny] where Nx and Ny are the output resolution of the raytraced file.
[-obj e n xt yt zt a b c r1 r2 r3 theta] where e and n are the superquadric 
    exponents, (xt, yt, zt) is a translation vector, (a, b, c) is a scaling
    vector, (r1, r2, r3) is the rotation axis, and theta is the rotation angle.
    This input will create a new superquadric object with these values.
[-ep val] where val is the epsilon value used for the Newton's update rule for
          checking if a value is close enough to zero.
[-bg r g b] where r, g, and b are the rgb values for the background color.
[-la x y z] where (x, y, z) is the position of the "Look At" point.
[-eye x y z] where (x, y, z) is the position of the Eye or "Look From" point.
[-f fd fw] where fd is the distance from the eye of the film plane and fw is the
           width of the film plane.
[-up x y z] where (x, y, z) is the film plane "Up" vector.
[-l x y z r g b k] where (x, y, z) is the position for a new light, (r, g, b) is
                   are the rgb value for the light, and k is the attenutation
                   factor (specifying even a single light will override the
                   default lights). Using this input multiple times will add
                   multiple different lights to the scene.
[-le r g b k] where (r, g, b) is the rgb value for the eye light and k is
              the attenuation factor (the position of this light will be set by
              the program at the position of the eye). This input can be
              specified multiple times, but only the inputs from the first time
              will be used
[-mat dr dg db sr sg sb shine refraction opacity] where (dr, dg, db) is the 
    diffuse rgb value, (sr, sg, sb) is the specular rgb value, shine is the 
    shininess value (higher number means more shininess), refraction is the
    snell ratio of the object (should be 1 or less), and opacity is the opacity
    of the object. (Note about opacity: 0 means completely solid, 1 means 
    completely see-through)
[-anti] toggles antialiasing (defaults to off). Beware, this will multiply
        runtime by a factor of 9.


All inputs require the first "dash" input to be typed exactly as you see it; the
following values are to be replaced with numeric inputs. All values default to
the assignment-specified defaults if they are not specified.

-obj, -mat, and -l can be specified multiple times to create multiple of each
item that they create. The rest of the commands can be specified multiple times
but will only use the last values that are specified (with the exception of -le,
which will only use the first values that are specified).

When specifying a rotation axis, do *not* give an axis of (0,0,0). This will
cause the program to never return...

Several text files have been included with some scene presets in the scenes 
directory; use them if you'd like. The image output that I produced for many of
the scenes is in the images directory, and they should be named respective to 
the .txt file that created them. However, if you aren't a fan of profanity, I 
suggest you stay away from objects4.txt and objects4Anti.txt. Aditionally, any
of the .txt files that include both anti-aliasing and a resolution of 800 x 800
either take an indeterminably long time to return, or don't return at all (this
applies mainly to objects10.txt).
