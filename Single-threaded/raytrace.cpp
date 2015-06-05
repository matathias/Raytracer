/* David Warrick
 * raytrace.cpp
 */

#include "util.h"
#include <GL/glew.h>
#include <GL/glut.h>
#include <math.h>
#define _USE_MATH_DEFINES
#include <float.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <stdexcept>

using namespace std;

struct Point_Light
{
    Vector3d position;
    Vector3d color;
    float attenuation_k;
};

struct Material
{
    Vector3d diffuse;
    Vector3d ambient;
    Vector3d specular;
    float shine;
    float snell;
    float opacity;
};

struct Object
{
    float e;
    float n;
    Material mat;
    MatrixXd scale;
    MatrixXd unScale;
    MatrixXd rotate;
    MatrixXd unRotate;
    Vector3d translate;
};

/******************************************************************************/
// Function prototypes

void init(void);
void initPPM();

double rad2deg(double);
double deg2rad(double);

void create_PPM_lights();
void create_film_plane();
void create_default_object();
Material get_default_material();

Vector3d cProduct(Vector3d a, Vector3d b);
Vector3d lighting(Vector3d point, Vector3d n, Vector3d dif, Vector3d amb, 
                  Vector3d spec, float shine, vector<Point_Light> l, Vector3d e,
                  int ind, int generation);
Vector3d findFilmA(float x, float y);
vector<MatrixXd> rayTrace();

void printPPM(int pixelIntensity, int xre, int yre, vector<MatrixXd> grid);
void parseArguments(int argc, char* argv[]);
void getArguments(int argc, char* argv[]);
void parseFile(char* filename);

/******************************************************************************/
// Global variables

// Tolerance value for the Newton's Method update rule
double epsilon = 0.00001;

// Toggle for using default object or objects loaded from input
bool defaultObject = true;
// Toggle for using default lights or lights loaded from input
bool defaultLights = true;
// Toggle for using antialiasing
bool antiAlias = false;

/* Ray-tracing globals */
// Unit orthogonal film vectors
Vector3d e1(1.0, 0.0, 0.0);
Vector3d e2(0.0, 1.0, 0.0);
Vector3d e3(0.0, 0.0, 1.0);

Vector3d lookAt(0.0, 0.0, 0.0);
Vector3d lookFrom(5.0, 5.0, 5.0);

Vector3d up(0.0, 1.0, 0.0);

Vector3d bgColor(0.0, 0.0, 0.0);

double filmDepth = 0.05;
double filmX = 0.035;
double filmY = 0.035;
int Nx = 100, Ny = 100;

vector<Point_Light> lightsPPM;
vector<Object> objects;

// Name of output file
string outName = "output/out.ppm";

/******************************************************************************/
// Function declarations

void initPPM()
{
    if (defaultObject)
        create_default_object();
    if (defaultLights)
        create_PPM_lights();

    create_film_plane();
}

// Component-wise vector3d product
Vector3d cProduct(Vector3d a, Vector3d b)
{
    Vector3d prod(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
    return prod;
}

// Phong reflection algorithm
Vector3d lighting(Vector3d point, Vector3d n, Vector3d dif, Vector3d amb, 
                  Vector3d spec, float shine, vector<Point_Light> l, Vector3d e,
                  int ind, int generation)
{
    Vector3d diffuseSum(0.0, 0.0, 0.0);
    Vector3d specularSum(0.0, 0.0, 0.0);
    Vector3d reflectedLight(0.0, 0.0, 0.0);
    Vector3d refractedLight(0.0, 0.0, 0.0);

    // Get the unit direction from the point to the camera
    Vector3d eDirection = e - point;
    eDirection.normalize();

    for (int i = 0; i < l.size() && generation > 0; i++)
    {
        // Retrieve the light's postion, color, and attenuation factor
        Vector3d lP = l[i].position;
        Vector3d lC = l[i].color;
        float attenuation = l[i].attenuation_k;

        // Get the unit direction and the distance between the light and the
        // point
        Vector3d lDirection = lP - point;
        float lightDist = lDirection.norm();
        lDirection.normalize();

        // Check to see that the light isn't blocked before considering it 
        // further. 
        // The i > 0 condition is present to prevent the program from blocking
        // anything from the eyelight, for the obvious reason that anything we
        // can see will be illuminated by the eyelight.
        bool useLight = true;
        for (int k = 0; k < objects.size() && useLight && i > 0; k++)
        {
            if (k != ind)
            {
                // Find the ray equation transformations
                Vector3d newA = newa(objects[k].unScale, objects[k].unRotate, 
                                     lDirection);
                Vector3d newB = newb(objects[k].unScale, objects[k].unRotate, 
                                     objects[k].translate * -1, point);

                // Find the quadratic equation coefficients
                Vector3d coeffs = findCoeffs(newA, newB, true);
                // Using the coefficients, find the roots
                Vector2d roots = findRoots(coeffs);

                // Check to see if the roots are FLT_MAX - if they are then the 
                // ray missed the superquadric. If they haven't missed then we 
                // can continue with the calculations.
                if (roots(0) != FLT_MAX)
                {
                    // Use the update rule to find tfinal
                    double tini = min(roots(0), roots(1));
                    double tfinal = updateRule(newA, newB, objects[k].e, 
                                                objects[k].n, tini, epsilon);

                    /* Check to see if tfinal is FLT_MAX - if it is then the ray 
                     * missed the superquadric. Additionally, if tfinal is 
                     * negative then either the ray has started inside the 
                     * object or is pointing away from the object; in both cases
                     * the ray has "missed". Also check to see if the object is
                     * farther away than the light - if it is then it isn't 
                     * actually blocking the light. */
                    double objDist = findRay(lDirection, point, tfinal).norm();
                    if (tfinal != FLT_MAX && tfinal >= 0 && objDist < lightDist)
                        useLight = false;
                }
            }
        }

        if (useLight)
        {
        
            // Find tthe attenuation term
            float atten = 1 / (float) (1 + (attenuation * pow(lightDist, 2)));
            // Add the attenuation factor to the light's color

            // Add the diffuse factor to the diffuse sum
            float nDotl = n.dot(lDirection);
            Vector3d lDiffuse = lC * atten * ((0 < nDotl) ? nDotl : 0);
            diffuseSum = diffuseSum + lDiffuse;

            // Add the specular factor to the specular sum
            Vector3d dirDif = eDirection + lDirection;
            dirDif.normalize();
            float nDotDir = n.dot(dirDif);
            Vector3d lSpecular = lC * atten * 
                         pow(((0 < nDotDir && 0 < nDotl) ? nDotDir : 0), shine);
            specularSum = specularSum + lSpecular;
        }
    }

    /* Find the light contribution from reflection */
    // Find the reflected ray
    Vector3d reflected = (2 * n * eDirection.dot(n)) - eDirection;
    reflected.normalize();
    double ttrueFinal = 0.0;
    int finalObj = 0;
    Vector3d finalNewA;
    Vector3d finalNewB;
    bool hitObject = false;
    for (int k = 0; k < objects.size() && generation > 0 ; k++)
    {
        if (k != ind)
        {
            // Find the ray equation transformations
            Vector3d newA = newa(objects[k].unScale, objects[k].unRotate, 
                                 reflected);
            Vector3d newB = newb(objects[k].unScale, objects[k].unRotate, 
                                 objects[k].translate * -1, point);

            // Find the quadratic equation coefficients
            Vector3d coeffs = findCoeffs(newA, newB, true);
            // Using the coefficients, find the roots
            Vector2d roots = findRoots(coeffs);

            // Check to see if the roots are FLT_MAX - if they are then the 
            // ray missed the superquadric. If they haven't missed then we 
            // can continue with the calculations.
            if (roots(0) != FLT_MAX)
            {
                // Use the update rule to find tfinal
                double tini = min(roots(0), roots(1));
                double tfinal = updateRule(newA, newB, objects[k].e, 
                                            objects[k].n, tini, epsilon);

                /* Check to see if tfinal is FLT_MAX - if it is then the ray 
                 * missed the superquadric. Additionally, if tfinal is negative 
                 * then either the ray has started inside the object or is 
                 * pointing away from the object; in both cases the ray has 
                 * "missed". */
                if (tfinal != FLT_MAX && tfinal >= 0)
                {
                    if(hitObject && tfinal < ttrueFinal)
                    {
                        ttrueFinal = tfinal;
                        finalObj = k;
                        finalNewA = newA;
                        finalNewB = newB;
                    }
                    else if (!hitObject)
                    {
                        hitObject = true;
                        ttrueFinal = tfinal;
                        finalObj = k;
                        finalNewA = newA;
                        finalNewB = newB;
                    }
                }
            }
        }
    }
    if (hitObject)
    {
        Vector3d intersectR = findRay(reflected, point, ttrueFinal);
        Vector3d intersectRNormal = unitNormal(objects[finalObj].rotate,
                                               finalNewA, finalNewB, ttrueFinal,
                                               objects[finalObj].e,
                                               objects[finalObj].n);
        reflectedLight = lighting(intersectR, intersectRNormal,
                                  objects[finalObj].mat.diffuse, 
                                  objects[finalObj].mat.ambient, 
                                  objects[finalObj].mat.specular, 
                                  objects[finalObj].mat.shine, 
                                  lightsPPM, lookFrom, finalObj, generation-1);
        if (shine < 1)
            reflectedLight = reflectedLight * shine;
    }
    

    /* Find the refraction contribution. */
    // Change the eye-direction vector so that it points at the surface instead
    // of at the eye
    eDirection = eDirection * -1;
    // Find the refracted ray
    Vector3d refracted1 = refractedRay(eDirection, n, objects[ind].mat.snell);
    refracted1.normalize();

    ttrueFinal = 0.0;
    finalObj = 0;
    hitObject = false;
    for (int k = 0; k < objects.size() && generation > 0; k++)
    {
        if (k != ind)
        {
            // Find the ray equation transformations
            Vector3d newA = newa(objects[k].unScale, objects[k].unRotate, 
                                 refracted1);
            Vector3d newB = newb(objects[k].unScale, objects[k].unRotate, 
                                 objects[k].translate * -1, point);

            // Find the quadratic equation coefficients
            Vector3d coeffs = findCoeffs(newA, newB, true);
            // Using the coefficients, find the roots
            Vector2d roots = findRoots(coeffs);

            // Check to see if the roots are FLT_MAX - if they are then the 
            // ray missed the superquadric. If they haven't missed then we 
            // can continue with the calculations.
            if (roots(0) != FLT_MAX)
            {
                // Use the update rule to find tfinal
                double tini = min(roots(0), roots(1));
                double tfinal = updateRule(newA, newB, objects[k].e, 
                                            objects[k].n, tini, epsilon);

                if (tfinal != FLT_MAX  && tfinal >= 0)
                {
                    if(hitObject && tfinal < ttrueFinal)
                    {
                        ttrueFinal = tfinal;
                        finalObj = k;
                        finalNewA = newA;
                        finalNewB = newB;
                    }
                    else if (!hitObject)
                    {
                        hitObject = true;
                        ttrueFinal = tfinal;
                        finalObj = k;
                        finalNewA = newA;
                        finalNewB = newB;
                    }
                }
            }
        }
    }
    if (hitObject)
    {
        Vector3d intersectR = findRay(refracted1, point, ttrueFinal);
        Vector3d intersectRNormal = unitNormal(objects[finalObj].rotate,
                                               finalNewA, finalNewB, ttrueFinal,
                                               objects[finalObj].e,
                                               objects[finalObj].n);

        refractedLight = objects[ind].mat.opacity * 
                         lighting(intersectR, intersectRNormal,
                                  objects[finalObj].mat.diffuse, 
                                  objects[finalObj].mat.ambient, 
                                  objects[finalObj].mat.specular, 
                                  objects[finalObj].mat.shine, 
                                  lightsPPM, lookFrom, finalObj, generation-1);
    }
    else
    {
        Vector3d refA = newa(objects[ind].unScale, objects[ind].unRotate, refracted1);
        Vector3d refB = newb(objects[ind].unScale, objects[ind].unRotate, 
                             objects[ind].translate * -1, point);
        Vector3d refCoeffs = findCoeffs(refA, refB, true);
        Vector2d refRoots = findRoots(refCoeffs);

        double tini = max(refRoots(0), refRoots(1));

        double tfinalRef = updateRule(refA, refB, objects[ind].e, 
                                       objects[ind].n, tini, epsilon);

        bool isRefracted = true;
        Vector3d outPoint, outNormal, outRay;
        if (isRefracted)
        {
            outPoint = findRay(refracted1, point, tfinalRef);
            outNormal = unitNormal(objects[ind].rotate, refA, refB, tfinalRef,
                                   objects[ind].e, objects[ind].n);
            outRay = refractedRay(refracted1, outNormal, 
                                  (double) 1 / objects[ind].mat.snell);
            // If the point has total internal reflection, then don't bother
            // with the rest of the refraction calculations.
            if(outRay(0) == FLT_MAX)
                isRefracted = false;
        }
        // Now that we've found where the ray exits, check to see if it hits any
        // objects; if it does, find the color contribution from that object
        ttrueFinal = 0.0;
        finalObj = 0;
        hitObject = false;
        for (int k = 0; k < objects.size() && generation > 0 && isRefracted; k++)
        {
            if (k != ind)
            {
                // Find the ray equation transformations
                Vector3d newA = newa(objects[k].unScale, objects[k].unRotate, 
                                     outRay);
                Vector3d newB = newb(objects[k].unScale, objects[k].unRotate, 
                                     objects[k].translate * -1, outPoint);

                // Find the quadratic equation coefficients
                Vector3d coeffs = findCoeffs(newA, newB, true);
                // Using the coefficients, find the roots
                Vector2d roots = findRoots(coeffs);

                // Check to see if the roots are FLT_MAX - if they are then the 
                // ray missed the superquadric. If they haven't missed then we 
                // can continue with the calculations.
                if (roots(0) != FLT_MAX)
                {
                    // Use the update rule to find tfinal
                    double tini = min(roots(0), roots(1));
                    double tfinal = updateRule(newA, newB, objects[k].e, 
                                                objects[k].n, tini, epsilon);

                    if (tfinal != FLT_MAX && tfinal >= 0)
                    {
                        if(hitObject && tfinal < ttrueFinal)
                        {
                            ttrueFinal = tfinal;
                            finalObj = k;
                            finalNewA = newA;
                            finalNewB = newB;
                        }
                        else if (!hitObject)
                        {
                            hitObject = true;
                            ttrueFinal = tfinal;
                            finalObj = k;
                            finalNewA = newA;
                            finalNewB = newB;
                        }
                    }
                }
            }
        }
        if (hitObject)
        {
            Vector3d intersectR = findRay(outRay, outPoint, ttrueFinal);
            Vector3d intersectRNormal = unitNormal(objects[finalObj].rotate,
                                                   finalNewA, finalNewB, 
                                                   ttrueFinal,
                                                   objects[finalObj].e,
                                                   objects[finalObj].n);

            refractedLight = objects[ind].mat.opacity * 
                             lighting(intersectR, intersectRNormal,
                                      objects[finalObj].mat.diffuse, 
                                      objects[finalObj].mat.ambient, 
                                      objects[finalObj].mat.specular, 
                                      objects[finalObj].mat.shine, 
                                      lightsPPM, lookFrom, finalObj, 
                                      generation - 1);
        }
    }

    Vector3d minVec(1,1,1);
    //Vector3d color = minVec.cwiseMin(amb + cProduct(diffuseSum, dif) +
    //                           cProduct(specularSum, spec));
    // No ambient term - is replaced by a light from the eye
    Vector3d color = minVec.cwiseMin(cProduct(diffuseSum, dif) +
                               cProduct(specularSum, spec) + 
                               reflectedLight + refractedLight);
    return color;
}

// helper function to find a point's location from film plane coordinates
Vector3d findFilmA(double x, double y)
{
    return (filmDepth * e3) + (x * e1) + (y * e2);
}

// Returns a vector of three Nx by Ny Matrices holding the color value for each
// pixel to display
vector<MatrixXd> rayTrace()
{
    vector<MatrixXd> output;
    MatrixXd red(Ny, Nx);
    MatrixXd green(Ny, Nx);
    MatrixXd blue(Ny, Nx);

    output.push_back(red);
    output.push_back(green);
    output.push_back(blue);
    
    double dx = filmX / (double) Nx;
    double dy = filmY / (double) Ny;

    double ttrueFinal = 0.0;
    int finalObj = 0;
    Vector3d finalNewA;
    Vector3d finalNewB;
    bool hitObject = false;
    
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            // The positions are subtracted by a Nx/2 or Ny/2 term to center
            // the film plane
            double px = (i * dx) - (filmX / (double) 2);
            double py = (j * dy) - (filmY / (double) 2);
            Vector3d pxColor(bgColor(0), bgColor(1), bgColor(2));
            if (!antiAlias)
            {
                Vector3d pointA = findFilmA(px, py);
                hitObject = false;
                finalObj = 0, ttrueFinal = 0;
                for (int k = 0; k < objects.size(); k++)
                {
                    // Find the ray equation transformations
                    Vector3d newA = newa(objects[k].unScale, objects[k].unRotate, pointA);
                    Vector3d newB = newb(objects[k].unScale, objects[k].unRotate, 
                                         objects[k].translate * -1, lookFrom);

                    // Find the quadratic equation coefficients
                    Vector3d coeffs = findCoeffs(newA, newB, true);
                    // Using the coefficients, find the roots
                    Vector2d roots = findRoots(coeffs);

                    // Check to see if the roots are FLT_MAX - if they are then the 
                    // ray missed the superquadric. If they haven't missed then we 
                    // can continue with the calculations.
                    if (roots(0) != FLT_MAX)
                    {
                        // Use the update rule to find tfinal
                        double tini = min(roots(0), roots(1));
                        double tfinal = updateRule(newA, newB, objects[k].e, 
                                                    objects[k].n, tini, epsilon);

                        /* Check to see if tfinal is FLT_MAX - if it is then the ray 
                         * missed the superquadric. Additionally, if tfinal is negative 
                         * then either the ray has started inside the object or is 
                         * pointing away from the object; in both cases the ray has 
                         * "missed". */
                        if (tfinal != FLT_MAX && tfinal >= 0)
                        {
                            if(hitObject && tfinal < ttrueFinal)
                            {
                                ttrueFinal = tfinal;
                                finalObj = k;
                                finalNewA = newA;
                                finalNewB = newB;
                            }
                            else if (!hitObject)
                            {
                                hitObject = true;
                                ttrueFinal = tfinal;
                                finalObj = k;
                                finalNewA = newA;
                                finalNewB = newB;
                            }
                        }
                    }
                }
                if(hitObject)
                {
                    Vector3d intersect = findRay(pointA, lookFrom, ttrueFinal);
                    Vector3d intersectNormal = unitNormal(objects[finalObj].rotate, 
                                                          finalNewA, finalNewB, 
                                                          ttrueFinal, 
                                                          objects[finalObj].e, 
                                                          objects[finalObj].n);

                    Vector3d color = lighting(intersect, intersectNormal,
                                              objects[finalObj].mat.diffuse, 
                                              objects[finalObj].mat.ambient, 
                                              objects[finalObj].mat.specular, 
                                              objects[finalObj].mat.shine, 
                                              lightsPPM, lookFrom, finalObj, 3);

                    pxColor = color;
                }
            }
            else
            {
                float denom = 3 + (2 / sqrt(2));
                vector<float> pxCoeffs;
                pxCoeffs.push_back((1 / (2 * sqrt(2))) / denom);
                pxCoeffs.push_back((1 / (float) 2) / denom);
                pxCoeffs.push_back((1 / (2 * sqrt(2))) / denom);
                pxCoeffs.push_back((1 / (float) 2) / denom);
                pxCoeffs.push_back(1 / denom);
                pxCoeffs.push_back((1 / (float) 2) / denom);
                pxCoeffs.push_back((1 / (2 * sqrt(2))) / denom);
                pxCoeffs.push_back((1 / (float) 2) / denom);
                pxCoeffs.push_back((1 / (2 * sqrt(2))) / denom);
                int counter = 0;
                for (int i = -1; i <= 1; i++)
                {
                    for (int j = -1; j <= 1; j++)
                    {
                        double thisPx = px + (i * (dx / (double) 2));
                        double thisPy = py + (j * (dy / (double) 2));
                        Vector3d pointA = findFilmA(thisPx, thisPy);
                        hitObject = false;
                        finalObj = 0, ttrueFinal = 0;
                        for (int k = 0; k < objects.size(); k++)
                        {
                            // Find the ray equation transformations
                            Vector3d newA = newa(objects[k].unScale, objects[k].unRotate, pointA);
                            Vector3d newB = newb(objects[k].unScale, objects[k].unRotate, 
                                                 objects[k].translate * -1, lookFrom);

                            // Find the quadratic equation coefficients
                            Vector3d coeffs = findCoeffs(newA, newB, true);
                            // Using the coefficients, find the roots
                            Vector2d roots = findRoots(coeffs);

                            // Check to see if the roots are FLT_MAX - if they are then the 
                            // ray missed the superquadric. If they haven't missed then we 
                            // can continue with the calculations.
                            if (roots(0) != FLT_MAX)
                            {
                                // Use the update rule to find tfinal
                                double tini = min(roots(0), roots(1));
                                double tfinal = updateRule(newA, newB, objects[k].e, 
                                                            objects[k].n, tini, epsilon);

                                /* Check to see if tfinal is FLT_MAX - if it is then the ray 
                                 * missed the superquadric. Additionally, if tfinal is negative 
                                 * then either the ray has started inside the object or is 
                                 * pointing away from the object; in both cases the ray has 
                                 * "missed". */
                                if (tfinal != FLT_MAX && tfinal >= 0)
                                {
                                    if(hitObject && tfinal < ttrueFinal)
                                    {
                                        ttrueFinal = tfinal;
                                        finalObj = k;
                                        finalNewA = newA;
                                        finalNewB = newB;
                                    }
                                    else if (!hitObject)
                                    {
                                        hitObject = true;
                                        ttrueFinal = tfinal;
                                        finalObj = k;
                                        finalNewA = newA;
                                        finalNewB = newB;
                                    }
                                }
                            }
                        }
                        if(hitObject)
                        {
                            Vector3d intersect = findRay(pointA, lookFrom, ttrueFinal);
                            Vector3d intersectNormal = unitNormal(objects[finalObj].rotate, 
                                                                  finalNewA, finalNewB, 
                                                                  ttrueFinal, 
                                                                  objects[finalObj].e, 
                                                                  objects[finalObj].n);

                            Vector3d color = lighting(intersect, intersectNormal,
                                                      objects[finalObj].mat.diffuse, 
                                                      objects[finalObj].mat.ambient, 
                                                      objects[finalObj].mat.specular, 
                                                      objects[finalObj].mat.shine, 
                                                      lightsPPM, lookFrom, finalObj, 3);

                            pxColor = pxColor + (color * pxCoeffs.at(counter));
                        }
                        counter++;
                    }
                }
                
            }
            output[0](j,i) = pxColor(0);
            output[1](j,i) = pxColor(1);
            output[2](j,i) = pxColor(2);
        }
    }

    return output;
}

void create_film_plane()
{
    // First, find the proper value for filmY from filmX, Nx, and Ny
    filmY = Ny * filmX / (float) Nx;

    // Find and set the plane vectors
    e3 = lookAt - lookFrom;
    e3.normalize();

    float alpha = up.dot(e3) / (float) e3.dot(e3);
    e2 = up - (alpha * e3);
    e2.normalize();

    e1 = e2.cross(e3);
    e1.normalize();
}

void create_default_object()
{
    Object obj;
    obj.e = 1.0;
    obj.n = 1.0;

    Vector3d trans(0.0, 0.0, 0.0);
    MatrixXd scal = matrix4to3(get_scale_mat(1.0, 1.0, 1.0));
    MatrixXd rot = matrix4to3(get_rotate_mat(0.0, 0.0, 1.0, 0.0));
    obj.scale = scal;
    obj.unScale = scal;
    obj.rotate = rot;
    obj.unRotate = rot;
    obj.translate = trans;

    obj.mat = get_default_material();

    objects.push_back(obj);
}

Material get_default_material()
{
    Vector3d dif(0.5, 0.5, 0.5);
    Vector3d amb(0.01, 0.01, 0.01);
    Vector3d spec(0.5, 0.5, 0.5);

    Material mat;
    mat.diffuse = dif;
    mat.ambient = amb;
    mat.specular = spec;
    mat.shine = 10;
    mat.snell = 0.9;
    mat.opacity = 0.01;

    return mat;
}

void create_PPM_lights()
{
    Point_Light eyeLight;
    eyeLight.position = lookFrom;
    eyeLight.color(0) = 1;
    eyeLight.color(1) = 1;
    eyeLight.color(2) = 1;
    eyeLight.attenuation_k = 0.01;
    lightsPPM.push_back(eyeLight);

    Point_Light light1;
    light1.position(0) = -10;
    light1.position(1) = 10;
    light1.position(2) = 10;

    light1.color(0) = 1;
    light1.color(1) = 1;
    light1.color(2) = 1;
    light1.attenuation_k = 0.001;

    lightsPPM.push_back(light1);
}

// Print pixel data to output
void printPPM(int pixelIntensity, int xre, int yre, vector<MatrixXd> grid)
{
    ofstream outFile;
    outFile.open(outName.c_str());
    
    // Print the PPM data to standard output
    outFile << "P3" << endl;
    outFile << xre << " " << yre << endl;
    outFile << pixelIntensity << endl;

    for (int j = 0; j < yre; j++)
    {
        for (int i = 0; i < xre; i++)
        {
            int red = grid[0](j,i) * pixelIntensity;
            int green = grid[1](j,i) * pixelIntensity;
            int blue = grid[2](j,i) * pixelIntensity;
            
            outFile << red << " " << green << " " << blue << endl;
        }
    }
    outFile.close();
}

// Function to parse the command line arguments
void parseArguments(int argc, char* argv[])
{
    int inInd = 1;
    
    // Command line triggers to respond to.
    const char* objectsIn = "-obj"; // the following values are: e, n, xt, yt, zt,
                                   // a, b, c, r1, r2, r3, theta
    const char* inMats = "-mat"; // the next several values are the diffuse rgb, 
                                  //specular rgb, shininess value, and refractive index of the 
                                  // object material
    const char* epsilonC = "-ep"; // the epsilon-close-to-zero value for the update rule
    const char* debugP = "-debug"; // tells the program to pring debugging values
    const char* background = "-bg"; // next three values are the rgb for the background
    const char* target = "-la"; // the next three values are the x,y,z for the look at vector
    const char* eye = "-eye"; // the next three values are the x,y,z for the look from vector
    const char* filmP = "-f"; // the next two values are the film plane depth and film plane width
    const char* nres = "-res"; // the next two valures are Nx and Ny
    const char* filmUp = "-up"; // the next three values are the x,y,z forr the film plane "up" vector
    const char* inLights = "-l"; // the next several values are the position, 
                                 //color, and attenuation coefficient for a new light
    const char* inEye = "-le"; // the next four values are the rgb and k for the eye light.
                                // only one eye light can be specified.
    const char* antiAliasing = "-anti"; // toggles antialiasing

    // Temporary values to store the in parameters. These only get assigned to
    // the actual program values if no errors are encountered while parsing the
    // arguments.
    vector<Object> tempObjs;
    vector<Material> tempMats;
    vector<Point_Light> tempLights;

    double tepsilon = epsilon;
    Vector3d tlookAt = lookAt, tlookFrom = lookFrom;
    Vector3d tup = up;
    Vector3d tbgColor = bgColor;
    double tfilmDepth = filmDepth;
    double tfilmX = filmX;
    int tNx = Nx, tNy = Ny;

    Vector3d eyeColor(1.0,1.0,1.0);
    double eyeK = 0.01;

    bool tdefaultObject = defaultObject;
    bool tdefaultLights = defaultLights;
    bool eyeSpecified = false;
    bool tantiAlias = antiAlias;

    try
    {
        while (inInd < argc)
        {
            if (strcmp(argv[inInd], objectsIn) == 0)
            {
                inInd += 12;
                if (inInd >= argc) 
                    throw out_of_range("Missing argument(s) for -obj [e n xt yt zt a b c r1 r2 r3 theta]");
                Object tobj;
                tobj.e = atof(argv[inInd-11]);
                tobj.n = atof(argv[inInd-10]);
                tobj.translate(0) = atof(argv[inInd-9]);
                tobj.translate(1) = atof(argv[inInd-8]);
                tobj.translate(2) = atof(argv[inInd-7]);
                float ta = atof(argv[inInd-6]);
                float tb = atof(argv[inInd-5]);
                float tc = atof(argv[inInd-4]);
                float tr1 = atof(argv[inInd-3]);
                float tr2 = atof(argv[inInd-2]);
                float tr3 = atof(argv[inInd-1]);
                float ttheta = atof(argv[inInd]);
                tobj.scale = matrix4to3(get_scale_mat(ta, tb, tc));
                tobj.unScale = matrix4to3(get_scale_mat(1 / (float) ta, 
                                                        1 / (float) tb, 
                                                        1 / (float) tc));
                tobj.rotate = matrix4to3(get_rotate_mat(tr1, tr2, tr3, ttheta));
                tobj.unRotate = matrix4to3(get_rotate_mat(tr1, tr2, tr3, -ttheta));
                tobj.mat = get_default_material();
                tempObjs.push_back(tobj);
                tdefaultObject = false;
            }
            else if (strcmp(argv[inInd], inMats) == 0)
            {
                inInd += 9;
                if (inInd >= argc) throw out_of_range("Missing argument(s) for -mat [dr dg db sr sg sb shine refraction opacity]");
                tdefaultObject = false;
                Material mat;
                mat.diffuse(0) = atof(argv[inInd-8]);
                mat.diffuse(1) = atof(argv[inInd-7]);
                mat.diffuse(2) = atof(argv[inInd-6]);
                mat.specular(0) = atof(argv[inInd-5]);
                mat.specular(1) = atof(argv[inInd-4]);
                mat.specular(2) = atof(argv[inInd-3]);
                mat.shine = atof(argv[inInd-2]);
                mat.snell = atof(argv[inInd-1]);
                mat.opacity = atof(argv[inInd]);
                mat.ambient(0) = 0, mat.ambient(1) = 0, mat.ambient(2) = 0;
                tempMats.push_back(mat);
            }
            else if (strcmp(argv[inInd], inLights) == 0)
            {
                inInd += 7;
                if (inInd >= argc) throw 
                                out_of_range("Missing argument(s) for -l [x y z r g b k]");
                tdefaultLights = false;
                Point_Light light;
                light.position(0) = atof(argv[inInd-6]);
                light.position(1) = atof(argv[inInd-5]);
                light.position(2) = atof(argv[inInd-4]);
                light.color(0) = atof(argv[inInd-3]);
                light.color(1) = atof(argv[inInd-2]);
                light.color(2) = atof(argv[inInd-1]);
                light.attenuation_k = atof(argv[inInd]);
                tempLights.push_back(light);
            }
            else if (strcmp(argv[inInd], inEye) == 0)
            {
                inInd += 4;
                if (inInd >= argc) throw out_of_range("Missing argument(s) for -le [r g b k]");
                if (!eyeSpecified)
                {
                    eyeColor(0) = atof(argv[inInd-3]);
                    eyeColor(1) = atof(argv[inInd-2]);
                    eyeColor(2) = atof(argv[inInd-1]);
                    eyeK = atof(argv[inInd]);
                    eyeSpecified = true;
                    tdefaultLights = false;
                }
            }
            else if (strcmp(argv[inInd], nres) == 0)
            {
                inInd += 2;
                if (inInd >= argc) throw out_of_range("Missing argument(s) for -res [Nx Ny]");
                tNx = atof(argv[inInd-1]);
                tNy = atof(argv[inInd]);
            }
            else if (strcmp(argv[inInd], filmP) == 0)
            {
                inInd += 2;
                if (inInd >= argc) throw out_of_range("Missing argument(s) for -f [Fd Fx]");
                tfilmDepth = atof(argv[inInd-1]);
                tfilmX = atof(argv[inInd]);
            }
            else if (strcmp(argv[inInd], epsilonC) == 0)
            {
                inInd++;
                if (inInd >= argc) throw out_of_range("Missing argument for -ep [epsilon]");
                tepsilon = atof(argv[inInd]);
            }
            else if (strcmp(argv[inInd], background) == 0)
            {
                inInd += 3;
                if (inInd >= argc) throw out_of_range("Missing argument(s) for -bg [x y z]");
                tbgColor(0) = atof(argv[inInd-2]);
                tbgColor(1) = atof(argv[inInd-1]);
                tbgColor(2) = atof(argv[inInd]);
            }
            else if (strcmp(argv[inInd], target) == 0)
            {
                inInd += 3;
                if (inInd >= argc) throw out_of_range("Missing argument(s) for -la [x y z]");
                tlookAt(0) = atof(argv[inInd-2]);
                tlookAt(1) = atof(argv[inInd-1]);
                tlookAt(2) = atof(argv[inInd]);
            }
            else if (strcmp(argv[inInd], eye) == 0)
            {
                inInd += 3;
                if (inInd >= argc) throw out_of_range("Missing argument(s) for -eye [x y z]");
                tlookFrom(0) = atof(argv[inInd-2]);
                tlookFrom(1) = atof(argv[inInd-1]);
                tlookFrom(2) = atof(argv[inInd]);
            }
            else if (strcmp(argv[inInd], filmUp) == 0)
            {
                inInd += 3;
                if (inInd >= argc) throw out_of_range("Missing argument(s) for -up [x y z]");
                tup(0) = atof(argv[inInd-2]);
                tup(1) = atof(argv[inInd-1]);
                tup(2) = atof(argv[inInd]);
            }
            else if (strcmp(argv[inInd], antiAliasing) == 0)
            {
                tantiAlias = true;
            }

            inInd++;
        }

        epsilon = tepsilon;
        lookAt = tlookAt, lookFrom = tlookFrom;
        up = tup;
        bgColor = tbgColor;
        filmDepth = tfilmDepth, filmX = tfilmX;
        Nx = tNx, Ny = tNy;
        defaultLights = tdefaultLights;
        defaultObject = tdefaultObject;
        antiAlias = tantiAlias;

        int i = 0;
        while (i < tempMats.size() &&  i < tempObjs.size())
        {
            tempObjs[i].mat = tempMats[i];
            i++;
        }

        objects = tempObjs;

        Point_Light eye;
        eye.position = lookFrom;
        eye.color = eyeColor;
        eye.attenuation_k = eyeK;

        vector<Point_Light>::iterator it;
        it = tempLights.begin();

        tempLights.insert(it, eye);

        if (!defaultLights)
            lightsPPM = tempLights;
    }
    catch (exception& ex)
    {
        cout << "Error at input argument " << inInd << ":" << endl;
        cout << ex.what() << endl;
        cout << "Program will execute with default values." << endl;
    }
}

void getArguments(int argc, char* argv[])
{
    if (argc > 1)
    {
        string filetype = ".txt";
        string firstArg(argv[1]);
        unsigned int isFile = firstArg.find(filetype);
        if ((int) isFile != (int) string::npos)
        {
            parseFile(argv[1]);
        }
        else
        {
            parseArguments(argc, argv);
        }
    }
}

void parseFile(char* filename)
{
    // Create an outfile name from the infile before parsing the file
    string inName(filename);
    printf("Input file name: %s\n", inName.c_str());
    // chop off the file extension
    unsigned int ext = inName.find(".txt");
    inName.erase(ext, 4);
    // if the file is in a directory path, chop off the directories
    // if your file system uses backslashes instead of forward slashes then eh,
    // I dunno. fuck you I guess.
    unsigned int delimiter = inName.find_last_of("/");
    if (delimiter != string::npos){
        inName.erase(0, delimiter+1);
    }
    outName = inName;
    outName.append(".ppm");
    outName.insert(0, "output/");
    printf("Output file name: %s\n\n", outName.c_str());
    ifstream ifs;
    ifs.open(filename);

    vector<char* > input;

    // Retrieve the data
    while(ifs.good())
    {
        // Read the next line
        string nextLine;
        getline(ifs, nextLine);

        while (nextLine.length() > 0)
        {
            // Get rid of extra spaces and read in any numbers that are
            // encountered
            string rotStr = " ";
            while (nextLine.length() > 0 && rotStr.compare(" ") == 0)
            {
                int space = nextLine.find(" ");
                if (space == 0)
                    space = 1;
                rotStr = nextLine.substr(0, space);
                nextLine.erase(0, space);
            }
            char* thistr = new char[rotStr.length() + 1];
            strcpy(thistr, rotStr.c_str());
            input.push_back(thistr);
        }
    }
    ifs.close();

    char* args[input.size()+1];
    for (int i = 0; i < input.size(); i++)
        args[i+1] = input.at(i);

    parseArguments(input.size()+1, args);
}

int main(int argc, char* argv[])
{
    // extract the command line arguments
    getArguments(argc, argv);

    initPPM();
    vector<MatrixXd> grid = rayTrace();

    printPPM(255, Nx, Ny, grid);
}

