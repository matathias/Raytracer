#include <cassert>
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include "raytraceCuda.cuh"

// flags as to whether or not reflection and refraction are included in the
// raytracing
#define REFLECTION 1
#define REFRACTION 1

#define SINGLETHREADMODE 0

#define RECURSIONDEPTH 3

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
    exit(code);
  }
}

struct Point_Light
{
    double position[3];    //3-vector
    double color[3];       //3-vector
    double attenuation_k;
};

struct Material
{
    double diffuse[3];     //3-vector
    double ambient[3];     //3-vector
    double specular[3];    //3-vector
    double shine;
    double snell;
    double opacity;
    double reflectivity;
};

struct Object
{
    double e;
    double n;
    Material mat;
    double scale[9];       //3x3-matrix
    double unScale[9];     //3x3-matrix
    double rotate[9];      //3x3-matrix
    double unRotate[9];    //3x3-matrix
    double translate[3];   //3-vector
    double unTranslate[3]; //3-vector
};
               
/********** Helper Functions **************************************************/
/* Stores the component-wise product of a and b into c. */
__device__
void cProduct(double *a, double *b, double *c) 
{
    c[0] = a[0] * b[0];
    c[1] = a[1] * b[1];
    c[2] = a[2] * b[2];
}

/* Stores the component-wise minimum of a and b into out. */
__device__
void cWiseMin(double *a, double *b, double *out)
{
    out[0] = min(a[0], b[0]);
    out[1] = min(a[1], b[1]);
    out[2] = min(a[2], b[2]);
}

/* Returns -1 for negative numbers, 1 for positive numbers, and 0 for zero. */
__device__
int sign(double s)
{
    if(s > 0) return 1;
    if(s < 0) return -1;
    return 0;
}

/* Returns the norm of the given vector. */
__device__
double d_norm(double *vec)
{
    return sqrt((vec[0] * vec[0]) + (vec[1] * vec[1]) + (vec[2] * vec[2]));
}

/* Normalizes the given vector. */
__device__
void d_normalize(double *vec)
{
    double n = d_norm(vec);
    vec[0] = vec[0] / (double) n;
    vec[1] = vec[1] / (double) n;
    vec[2] = vec[2] / (double) n;
}

/* Returns the dot product of the given vectors. */
__device__
double d_dot(double *a, double *b)
{
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
}

/* Implicit Superquadric function. */
// vec is a 3-vector
__device__
double isq(double *vec, double *e, double *n)
{
    // Test for n = 0 now to prevent divide-by-zero errors.
    if (n == 0)
        return FLT_MAX;
    
    double zTerm = pow(pow(vec[2], 2.0), 1.0 / (double) *n);

    // Test for e = 0 now to prevent divide-by-zero errors.
    if (e == 0)
        return zTerm;
        
    double xTerm = pow(pow(vec[0], 2.0), 1.0 / (double) *e);
    double yTerm = pow(pow(vec[1], 2.0), 1.0 / (double) *e);
    double xyTerm = pow(xTerm + yTerm, *e / (double) *n);
    return xyTerm + zTerm - 1.0;
}

/* Ray Equation */
// a and b are both 3-vectors
__device__
void findRay(double *a, double *b, double *c, double t)
{
    c[0] = (a[0] * t) + b[0];
    c[1] = (a[1] * t) + b[1];
    c[2] = (a[2] * t) + b[2];
}

/* Apply the Inverse Transform to a to get a new, usable a. */
// unScale and unRotate are 3x3 matrices. a and newA are 3-vectors
__device__
void newa(double *unScale, double *unRotate, double *a, double *newA)
{
    double a0 = (unRotate[0] * a[0]) + (unRotate[1] * a[1]) + (unRotate[2] * a[2]);
    double a1 = (unRotate[3] * a[0]) + (unRotate[4] * a[1]) + (unRotate[5] * a[2]);
    double a2 = (unRotate[6] * a[0]) + (unRotate[7] * a[1]) + (unRotate[8] * a[2]);
    
    newA[0] = (unScale[0] * a0) + (unScale[1] * a1) + (unScale[2] * a2);
    newA[1] = (unScale[3] * a0) + (unScale[4] * a1) + (unScale[5] * a2);
    newA[2] = (unScale[6] * a0) + (unScale[7] * a1) + (unScale[8] * a2);
}

/* Apply the Inverse Transform to b to get a new, usable b. */
// unScale and unRotate are 3x3 matrices. unTranslate, b, and newB are 3-vectors
__device__
void newb(double *unScale, double *unRotate, double *unTranslate, double *b, 
          double *newB)
{
    // b + unTranslate
    double b0 = b[0] + unTranslate[0];
    double b1 = b[1] + unTranslate[1];
    double b2 = b[2] + unTranslate[2];
    
    // unRotate * (b + unTranslate)
    newB[0] = (unRotate[0] * b0) + (unRotate[1] * b1) + (unRotate[2] * b2);
    newB[1] = (unRotate[3] * b0) + (unRotate[4] * b1) + (unRotate[5] * b2);
    newB[2] = (unRotate[6] * b0) + (unRotate[7] * b1) + (unRotate[8] * b2);
    
    b0 = newB[0];
    b1 = newB[1];
    b2 = newB[2];
    
    // unScale * (unRotate * (b + unTranslate))
    newB[0] = (unScale[0] * b0) + (unScale[1] * b1) + (unScale[2] * b2);
    newB[1] = (unScale[3] * b0) + (unScale[4] * b1) + (unScale[5] * b2);
    newB[2] = (unScale[6] * b0) + (unScale[7] * b1) + (unScale[8] * b2);
}

/* Finds the scalar coefficients of the quadratic equation with the two given
 * vectors. If positiveb is true then the returned coeffs will all be multiplied
 * by -1 if b is negative, to ensure that b is positive. */
// a, b, and c are 3-vectors
__device__
void findCoeffs(double *a, double *b, double *c, bool positiveb)
{
    c[0] = d_dot(a, a);
    c[1] = 2 * d_dot(a, b);
    c[2] = d_dot(b, b) - 3;
    
    if (positiveb && c[1] < 0){
        c[0] *= -1;
        c[1] *= -1;
        c[2] *= -1;
    }
}

/* Finds the roots of the quadratic with the coefficients specified by the input
 * Vector3d. If one of the roots is complex then FLT_MAX is returned instead. */
// coeffs is a 3-vector, roots is a 2-vector
__device__
void findRoots(double *coeffs, double *roots)
{
    double interior = pow(coeffs[1], 2) - (4 * coeffs[0] * coeffs[2]);
    if (interior < 0)
    {
        roots[0] = FLT_MAX;
        roots[1] = FLT_MAX;
    }
    else
    {
        roots[0] = (-coeffs[1] - sqrt(interior)) / (double) (2 * coeffs[0]);
        roots[1] = (2 * coeffs[2]) / (double) (-coeffs[1] - sqrt(interior));
    }
}

/* Gradient of the isq function. */
// vec and grad are 3-vectors
__device__
void isqGradient(double *vec, double *grad, double e, double n)
{
    double xval = 0.0, yval = 0.0, zval = 0.0;
    // Check for n = 0 to prevent divide-by-zero errors
    if (n == 0)
    {
        xval = yval = zval = FLT_MAX;
    }
    // Check for e = 0 to prevent divide-by-zero errors
    else if (e == 0)
    {
        xval = yval = FLT_MAX;
        zval = (2 * vec[2] * pow(pow(vec[2], 2), ((double) 1 / n) - 1)) / (double) n;
    }
    else
    {
        double xterm = pow(pow(vec[0], 2.0), (double) 1 / e);
        double yterm = pow(pow(vec[1], 2.0), (double) 1 / e);
        double xyterm = pow(xterm + yterm, ((double) e / n) - 1);
        double x2term = (2 * vec[0] * pow(pow(vec[0], 2.0), ((double) 1 / e) - 1));
        double y2term = (2 * vec[1] * pow(pow(vec[1], 2.0), ((double) 1 / e) - 1));
        xval = x2term * xyterm / (double) n;
        yval = y2term * xyterm / (double) n;
        zval = (2 * vec[2] * pow(pow(vec[2], 2.0), ((double) 1 / n) - 1)) / (double) n;
    }
    
    grad[0] = xval;
    grad[1] = yval;
    grad[2] = zval;
}

/* Derivative of the isq function. */
// vec and a are 3-vectors
__device__
double gPrime(double *vec, double *a, double e, double n)
{
    double tmp[3];
    isqGradient(vec, &tmp[0], e, n);
    double val = d_dot(a, &tmp[0]);
    return val;
}

/* Uses Newton's method to find the t value at which a ray hits the superquadric.
 * If the ray actually misses the superquadric then FLT_MAX is returned instead.*/
// a and b are 3-vectors
__device__
double updateRule(double *a, double *b, double *e, double *n, double t, double epsilon)
{
    double vec[3];
    
    findRay(a, b, &vec[0], t);
    double gP = gPrime(&vec[0], a, *e, *n);
    double gPPrevious = gP;
    double g = 0.0;
    double tnew = t, told = t;
    bool stopPoint = false;

    while (!stopPoint)
    {
        told = tnew;
        findRay(a, b, &vec[0], told);
        gP = gPrime(&vec[0], a, *e, *n);
        g = isq(&vec[0], e, n);

        if ((g - epsilon) <= 0)
        {
            stopPoint = true;
        }
        else if (sign(gP) != sign(gPPrevious) || gP == 0)
        {
            stopPoint = true;
            tnew = FLT_MAX;
        }
        else
        {
            tnew = told - (g / gP);
            gPPrevious = gP;
        }
    }

    return tnew;
}



/* Unit normal vector at a point on the superquadric */
// r is a 3x3 matrix
// vec1, vec2, and un are 3-vectors
__device__
void unitNormal(double *r, double *vec1, double *vec2, double *un, double tt, double e, double n)
{
    findRay(vec1, vec2, un, tt);
    isqGradient(un, un, e, n);
    
    double un0 = un[0];
    double un1 = un[1];
    double un2 = un[2];
    
    un[0] = (r[0] * un0) + (r[1] * un1) + (r[2] * un2);
    un[1] = (r[3] * un0) + (r[4] * un1) + (r[5] * un2);
    un[2] = (r[6] * un0) + (r[7] * un1) + (r[8] * un2);
    
    d_normalize(un);
}

// Returns the angle between two vectors.
// Both a and b are 3-vectors.
__device__
double vectorAngle(double *a, double *b)
{
    double d = d_dot(a, b);
    double mag = d_norm(a) * d_norm(b);

    return acos(d / (double) mag);
}

// Calculates the refracted ray from an input ray and normal and a snell ratio
// If there is total internal reflection, then a vector of FLT_MAX is returned
// instead.
// a, n, and ref are 3-vectors
__device__
void refractedRay(double *a, double *n, double *ref, double snell)
{
    double tmp = d_dot(n, a);
    n[0] *= -1;
    n[1] *= -1;
    n[2] *= -1;
    double cos1 = d_dot(n, a);
    if (cos1 < 0)
    {
        cos1 = tmp;
    }
    else {
        n[0] *= -1;
        n[1] *= -1;
        n[2] *= -1;
    }
    double radicand = 1 - (pow(snell, 2) * (1 - pow(cos1,2)));

    if (radicand < 0)
    {
        ref[0] = FLT_MAX;
        ref[1] = FLT_MAX;
        ref[2] = FLT_MAX;
    }
    else
    {
        double cos2 = sqrt(radicand);

        ref[0] = (snell * a[0]) + (((snell * cos1) - cos2) * n[0]);
        ref[1] = (snell * a[1]) + (((snell * cos1) - cos2) * n[1]);
        ref[2] = (snell * a[2]) + (((snell * cos1) - cos2) * n[2]);
    }
}

/* debugging purposes */
__device__
void print_objects(Object *p_objects, int numObjects)
{
    for (int i = 0; i < numObjects; i++) {
        Object *o = &p_objects[i];
        printf("\nObject %d\n", i);
        printf("e: %f\t n: %f\n", o->e, o->n);
        printf("scale: [%f, %f, %f] unScale: [%f, %f, %f]\n", o->scale[0],
               o->scale[1], o->scale[2], o->unScale[0], o->unScale[1], o->unScale[2]);
        printf("       [%f, %f, %f]          [%f, %f, %f]\n", o->scale[3],
               o->scale[4], o->scale[5], o->unScale[3], o->unScale[4], o->unScale[5]);
        printf("       [%f, %f, %f]          [%f, %f, %f]\n", o->scale[6],
               o->scale[7], o->scale[8], o->unScale[6], o->unScale[7], o->unScale[8]);
        printf("rotate: [%f, %f, %f] unRotate: [%f, %f, %f]\n", o->rotate[0],
               o->rotate[1], o->rotate[2], o->unRotate[0], o->unRotate[1], o->unRotate[2]);
        printf("        [%f, %f, %f]           [%f, %f, %f]\n", o->rotate[3],
               o->rotate[4], o->rotate[5], o->unRotate[3], o->unRotate[4], o->unRotate[5]);
        printf("        [%f, %f, %f]           [%f, %f, %f]\n", o->rotate[6],
               o->rotate[7], o->rotate[8], o->unRotate[6], o->unRotate[7], o->unRotate[8]);
        printf("translate: (%f, %f, %f) unTranslate: (%f, %f, %f)\n",
               o->translate[0], o->translate[1], o->translate[2], o->unTranslate[0],
               o->unTranslate[1], o->unTranslate[2]);
        printf("Material-\n");
        printf("Diffuse: (%f, %f, %f)\n", o->mat.diffuse[0], o->mat.diffuse[1],
               o->mat.diffuse[2]);
        printf("Ambient: (%f, %f, %f)\n", o->mat.ambient[0], o->mat.ambient[1],
               o->mat.ambient[2]);
        printf("Specular: (%f, %f, %f)\n", o->mat.specular[0], o->mat.specular[1],
               o->mat.specular[2]);
        printf("shine: %f\t snell: %f\t opacity: %f\n", o->mat.shine, o->mat.snell, o->mat.opacity);
    }
}
__device__
void print_lights(Point_Light *p_lights, int numLights)
{
    for (int i = 0; i < numLights; i++) {
        Point_Light *l = &p_lights[i];
        printf("\nLight %d\n", i);
        printf("Position: (%f, %f, %f)\n", l->position[0], l->position[1], l->position[2]);
        printf("Color: (%f, %f, %f)\n", l->color[0], l->color[1], l->color[2]);
        printf("Attenuation Factor: %f\n", l->attenuation_k);
    }
}

/********** Actual Raytracing Functions ***************************************/
__device__
// n is the normal. e is the eye. ind is the index of the object we're lighting.
void lighting(double *point, double *n, double *e, Material *mat,
              Point_Light *l, int numLights, 
              Object *objects, int numObjects,
              double epsilon, int ind, int generation, double *res,
              double *lightDoubles)
{
    double diffuseSum[3] = {0.0, 0.0, 0.0};
    double specularSum[3] = {0.0, 0.0, 0.0};
    double refractedLight[3] = {0.0, 0.0, 0.0};
    double reflectedLight[3] = {0.0, 0.0, 0.0};
    
    double* dif = &mat->diffuse[0];
    double* spec = &mat->specular[0];
    double shine = mat->shine;
    
    double *newA = &lightDoubles[0];
    double *newB = &lightDoubles[3];
    double *coeffs = &lightDoubles[6];
    double *roots = &lightDoubles[30];
    

    // Get the unit direction from the point to the camera
    double eDirection[3];
    
    for (int i = 0; i < 3; i++)
        eDirection[i] = e[i] - point[i];
        
    d_normalize(&eDirection[0]);

    for (int i = 0; i < numLights && generation > 0; i++)
    {
        // Retrieve the light's postion, color, and attenuation factor
        double attenuation = l[i].attenuation_k;

        // Get the unit direction and the distance between the light and the
        // point
        double lDirection[3];
        
        lDirection[0] = l[i].position[0] - point[0];
        lDirection[1] = l[i].position[1] - point[1];
        lDirection[2] = l[i].position[2] - point[2];
            
        double lightDist = d_norm(&lDirection[0]);
        d_normalize(&lDirection[0]);

        // Check to see that the light isn't blocked before considering it 
        // further. 
        // The i > 0 condition is present to prevent the program from blocking
        // anything from the eyelight, for the obvious reason that anything we
        // can see will be illuminated by the eyelight.
        bool useLight = true;
        for (int k = 0; k < numObjects && useLight && i > 0; k++)
        {
            if (k != ind)
            {
                // Find the ray equation transformations
                newa(&objects[k].unScale[0], &objects[k].unRotate[0], 
                     &lDirection[0], &newA[0]);
                newb(&objects[k].unScale[0], &objects[k].unRotate[0], 
                     &objects[k].unTranslate[0], point, &newB[0]);

                // Find the quadratic equation coefficients
                findCoeffs(&newA[0], &newB[0], &coeffs[0], true);
                // Using the coefficients, find the roots
                findRoots(&coeffs[0], &roots[0]);

                // Check to see if the roots are FLT_MAX - if they are then the 
                // ray missed the superquadric. If they haven't missed then we 
                // can continue with the calculations.
                if (roots[0] != FLT_MAX)
                {
                    // Use the update rule to find tfinal
                    double tini = min(roots[0], roots[1]);
                    double tfinal = updateRule(&newA[0], &newB[0], &objects[k].e, 
                                               &objects[k].n, tini, epsilon);

                    /* Check to see if tfinal is FLT_MAX - if it is then the ray 
                     * missed the superquadric. Additionally, if tfinal is 
                     * negative then either the ray has started inside the 
                     * object or is pointing away from the object; in both cases
                     * the ray has "missed". Also check to see if the object is
                     * farther away than the light - if it is then it isn't 
                     * actually blocking the light. */
                    double ray[3];
                    
                    findRay(&lDirection[0], point, &ray[0], tfinal);
                    double objDist = d_norm(&ray[0]);
                    if (tfinal != FLT_MAX && tfinal >= 0 && objDist < lightDist)
                        useLight = false;
                }
            }
        }

        if (useLight)
        {
        
            // Find tthe attenuation term
            double atten = 1 / (double) (1 + (attenuation * pow(lightDist, 2)));
            // Add the attenuation factor to the light's color

            // Add the diffuse factor to the diffuse sum
            double nDotl = d_dot(n, &lDirection[0]);
            //Vector3d lDiffuse = lC * atten * ((0 < nDotl) ? nDotl : 0);
            //diffuseSum = diffuseSum + lDiffuse;
            if (0 < nDotl) {
                diffuseSum[0] += l[i].color[0] * atten * nDotl;
                diffuseSum[1] += l[i].color[1] * atten * nDotl;
                diffuseSum[2] += l[i].color[2] * atten * nDotl;
            }

            // Add the specular factor to the specular sum
            double dirDif[3];
            
            dirDif[0] = eDirection[0] + lDirection[0];
            dirDif[1] = eDirection[1] + lDirection[1];
            dirDif[2] = eDirection[2] + lDirection[2];
            d_normalize(&dirDif[0]);
            double nDotDir = d_dot(n, &dirDif[0]);
            //Vector3d lSpecular = lC * atten * 
            //             pow(((0 < nDotDir && 0 < nDotl) ? nDotDir : 0), shine);
            //specularSum = specularSum + lSpecular;
            if (0 < nDotDir && 0 < nDotl) {
                specularSum[0] += l[i].color[0] * atten * pow(nDotDir, shine);
                specularSum[1] += l[i].color[1] * atten * pow(nDotDir, shine);
                specularSum[2] += l[i].color[2] * atten * pow(nDotDir, shine);
            }
        }
    }
    /* Find the light contribution from reflection */
    // Find the reflected ray
    
#if REFLECTION
    double ttrueFinal = 0.0;
    int finalObj = 0;
    bool hitObject = false;
    
    double *finalNewA = &lightDoubles[12];
    double *finalNewB = &lightDoubles[15];
    // If the object's reflectivity is 0 then just don't bother
    if (objects[ind].mat.reflectivity != 0) {
        double eDotN = d_dot(n, &eDirection[0]);
        double *reflected = &lightDoubles[9];
        reflected[0] = (2 * n[0] * eDotN) - eDirection[0];
        reflected[1] = (2 * n[1] * eDotN) - eDirection[1];
        reflected[2] = (2 * n[2] * eDotN) - eDirection[2];
        
        d_normalize(&reflected[0]);
        
        for (int k = 0; k < numObjects && generation > 0 ; k++)
        {
            if (k != ind)
            {
                // Find the ray equation transformations
                newa(objects[k].unScale, objects[k].unRotate, &reflected[0], 
                     &newA[0]);
                newb(objects[k].unScale, objects[k].unRotate, 
                     objects[k].unTranslate, point, &newB[0]);

                // Find the quadratic equation coefficients
                findCoeffs(&newA[0], &newB[0], &coeffs[0], true);
                // Using the coefficients, find the roots
                findRoots(&coeffs[0], &roots[0]);

                // Check to see if the roots are FLT_MAX - if they are then the 
                // ray missed the superquadric. If they haven't missed then we 
                // can continue with the calculations.
                if (roots[0] != FLT_MAX)
                {
                    // Use the update rule to find tfinal
                    double tini = min(roots[0], roots[1]);
                    double tfinal = updateRule(&newA[0], &newB[0], &objects[k].e, 
                                               &objects[k].n, tini, epsilon);

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
                            finalNewA[0] = newA[0];
                            finalNewA[1] = newA[1];
                            finalNewA[2] = newA[2];
                            finalNewB[0] = newB[0];
                            finalNewB[1] = newB[1];
                            finalNewB[2] = newB[2];
                        }
                        else if (!hitObject)
                        {
                            hitObject = true;
                            ttrueFinal = tfinal;
                            finalObj = k;
                            finalNewA[0] = newA[0];
                            finalNewA[1] = newA[1];
                            finalNewA[2] = newA[2];
                            finalNewB[0] = newB[0];
                            finalNewB[1] = newB[1];
                            finalNewB[2] = newB[2];
                        }
                    }
                }
            }
        }
        if (hitObject)
        {
            double intersectR[3];
            double intersectRNormal[3];
            
            findRay(&reflected[0], point, &intersectR[0], ttrueFinal);
            unitNormal(objects[finalObj].rotate, &finalNewA[0], &finalNewB[0], 
                       &intersectRNormal[0], ttrueFinal, objects[finalObj].e,
                       objects[finalObj].n);
                       
            lighting(&intersectR[0], &intersectRNormal[0], e,
                     &objects[finalObj].mat,
                     l, numLights, objects, numObjects, epsilon,
                     finalObj, generation-1, &reflectedLight[0], lightDoubles);
            
            // Multiply by the object's reflectivity         
            reflectedLight[0] *= objects[ind].mat.reflectivity;
            reflectedLight[1] *= objects[ind].mat.reflectivity;
            reflectedLight[2] *= objects[ind].mat.reflectivity;
        }
    }
#endif
    
#if REFRACTION
    /* Find the refraction contribution. */
    // If the object's opacity is zero then just don't bother
    if (objects[ind].mat.opacity != 0) {
        // Change the eye-direction vector so that it points at the surface instead
        // of at the eye
        eDirection[0] *= -1;
        eDirection[1] *= -1;
        eDirection[2] *= -1;
        // Find the refracted ray
        double *refracted1 = &lightDoubles[9];
        refractedRay(&eDirection[0], n, &refracted1[0], objects[ind].mat.snell);
        d_normalize(&refracted1[0]);

        ttrueFinal = 0.0;
        finalObj = 0;
        hitObject = false;
        for (int k = 0; k < numObjects && generation > 0; k++)
        {
            if (k != ind)
            {
                // Find the ray equation transformations
                newa(objects[k].unScale, objects[k].unRotate, &refracted1[0], &newA[0]);
                newb(objects[k].unScale, objects[k].unRotate, 
                     objects[k].unTranslate, point, &newB[0]);

                // Find the quadratic equation coefficients
                findCoeffs(&newA[0], &newB[0], &coeffs[0], true);
                // Using the coefficients, find the roots
                findRoots(&coeffs[0], &roots[0]);

                // Check to see if the roots are FLT_MAX - if they are then the 
                // ray missed the superquadric. If they haven't missed then we 
                // can continue with the calculations.
                if (roots[0] != FLT_MAX)
                {
                    // Use the update rule to find tfinal
                    double tini = min(roots[0], roots[1]);
                    double tfinal = updateRule(&newA[0], &newB[0], &objects[k].e, 
                                               &objects[k].n, tini, epsilon);

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
                            finalNewA[0] = newA[0];
                            finalNewA[1] = newA[1];
                            finalNewA[2] = newA[2];
                            finalNewB[0] = newB[0];
                            finalNewB[1] = newB[1];
                            finalNewB[2] = newB[2];
                        }
                        else if (!hitObject)
                        {
                            hitObject = true;
                            ttrueFinal = tfinal;
                            finalObj = k;
                            finalNewA[0] = newA[0];
                            finalNewA[1] = newA[1];
                            finalNewA[2] = newA[2];
                            finalNewB[0] = newB[0];
                            finalNewB[1] = newB[1];
                            finalNewB[2] = newB[2];
                        }
                    }
                }
            }
        }
        if (hitObject)
        {
            double intersectR[3];
            double intersectRNormal[3];
            
            findRay(&refracted1[0], point, &intersectR[0], ttrueFinal);
            unitNormal(objects[finalObj].rotate, &finalNewA[0], &finalNewB[0], 
                       &intersectRNormal[0], ttrueFinal, objects[finalObj].e,
                       objects[finalObj].n);

            lighting(&intersectR[0], &intersectRNormal[0], e,
                     &objects[finalObj].mat,
                     l, numLights, objects, numObjects, epsilon,
                     finalObj, generation-1, &refractedLight[0], lightDoubles);
            refractedLight[0] *= objects[ind].mat.opacity;
            refractedLight[1] *= objects[ind].mat.opacity;
            refractedLight[2] *= objects[ind].mat.opacity;
        }
        else
        {
            double *refA = &lightDoubles[18];
            double *refB = &lightDoubles[21];
            double *refCoeffs = &lightDoubles[24];
            double *refRoots = &lightDoubles[27];
            newa(objects[ind].unScale, objects[ind].unRotate, &refracted1[0], &refA[0]);
            newb(objects[ind].unScale, objects[ind].unRotate, 
                 objects[ind].unTranslate, point, &refB[0]);
            findCoeffs(&refA[0], &refB[0], &refCoeffs[0], true);
            findRoots(&refCoeffs[0], &refRoots[0]);

            double tini = max(refRoots[0], refRoots[1]);

            double tfinalRef = updateRule(&refA[0], &refB[0], &objects[ind].e, 
                                          &objects[ind].n, tini, epsilon);

            bool isRefracted = true;
            double outNormal[3];
            double *outPoint = &lightDoubles[24];
            double *outRay = &lightDoubles[27];
            
            if (isRefracted) // the fuck is the point of this?
            {
                findRay(&refracted1[0], point, &outPoint[0], tfinalRef);
                unitNormal(objects[ind].rotate, &refA[0], &refB[0], &outNormal[0], tfinalRef,
                           objects[ind].e, objects[ind].n);
                refractedRay(&refracted1[0], &outNormal[0], &outRay[0],
                             (double) 1 / objects[ind].mat.snell);
                // If the point has total internal reflection, then don't bother
                // with the rest of the refraction calculations.
                if(outRay[0] == FLT_MAX)
                    isRefracted = false;
            }
            // Now that we've found where the ray exits, check to see if it hits any
            // objects; if it does, find the color contribution from that object
            ttrueFinal = 0.0;
            finalObj = 0;
            hitObject = false;
            for (int k = 0; k < numObjects && generation > 0 && isRefracted; k++)
            {
                if (k != ind)
                {
                    // Find the ray equation transformations
                    newa(objects[k].unScale, objects[k].unRotate, 
                         &outRay[0], &newA[0]);
                    newb(objects[k].unScale, objects[k].unRotate, 
                         objects[k].unTranslate, &outPoint[0], &newB[0]);

                    // Find the quadratic equation coefficients
                    findCoeffs(&newA[0], &newB[0], &coeffs[0], true);
                    // Using the coefficients, find the roots
                    findRoots(&coeffs[0], &roots[0]);

                    // Check to see if the roots are FLT_MAX - if they are then the 
                    // ray missed the superquadric. If they haven't missed then we 
                    // can continue with the calculations.
                    if (roots[0] != FLT_MAX)
                    {
                        // Use the update rule to find tfinal
                        double tini = min(roots[0], roots[1]);
                        double tfinal = updateRule(&newA[0], &newB[0], &objects[k].e, 
                                                   &objects[k].n, tini, epsilon);

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
                                finalNewA[0] = newA[0];
                                finalNewA[1] = newA[1];
                                finalNewA[2] = newA[2];
                                finalNewB[0] = newB[0];
                                finalNewB[1] = newB[1];
                                finalNewB[2] = newB[2];
                            }
                            else if (!hitObject)
                            {
                                hitObject = true;
                                ttrueFinal = tfinal;
                                finalObj = k;
                                finalNewA[0] = newA[0];
                                finalNewA[1] = newA[1];
                                finalNewA[2] = newA[2];
                                finalNewB[0] = newB[0];
                                finalNewB[1] = newB[1];
                                finalNewB[2] = newB[2];
                            }
                        }
                    }
                }
            }
            if (hitObject)
            {
                double intersectR[3];
                double intersectRNormal[3];
                
                findRay(&outRay[0], &outPoint[0], &intersectR[0], ttrueFinal);
                unitNormal(objects[finalObj].rotate, &finalNewA[0], &finalNewB[0], 
                           &intersectRNormal[0], ttrueFinal, objects[finalObj].e,
                           objects[finalObj].n);

                lighting(&intersectR[0], &intersectRNormal[0], e,
                         &objects[finalObj].mat,
                         l, numLights, objects, numObjects, epsilon,
                         finalObj, generation - 1, &refractedLight[0], lightDoubles);
                refractedLight[0] *= objects[ind].mat.opacity;
                refractedLight[1] *= objects[ind].mat.opacity;
                refractedLight[2] *= objects[ind].mat.opacity;
            }
        }
    }
#endif

    double *minVec = &lightDoubles[0];
    double *maxVec = &lightDoubles[3];
    minVec[0] = 1;
    minVec[1] = 1;
    minVec[2] = 1;
           
    cProduct(&diffuseSum[0], dif, &diffuseSum[0]);
    cProduct(&specularSum[0], spec, &specularSum[0]);
    maxVec[0] = diffuseSum[0] + specularSum[0] + reflectedLight[0] + refractedLight[0];
    maxVec[1] = diffuseSum[1] + specularSum[1] + reflectedLight[1] + refractedLight[1];
    maxVec[2] = diffuseSum[2] + specularSum[2] + reflectedLight[2] + refractedLight[2];
    cWiseMin(&minVec[0], &maxVec[0], res);
}

__global__
void raytraceKernel(double *grid, Object *objects, Point_Light *lightsPPM,
                    double *data, double *bgColor, double *e1, double *e2,
                    double *e3, double *lookFrom, double *rayDoubles,
                    double *lightDoubles,
                    int Nx, int Ny, bool antiAliased)
{   
    /* data[0] = numObjects
     * data[1] = numLights
     * data[2] = filmX
     * data[3] = filmY
     * data[4] = epsilon
     * data[5] = filmDepth
     */
    // Parallize by screen pixel
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    
    // Debugging
    /*if (i == 0 && j == 0) {
        print_objects(objects, data[0]);
        print_lights(lightsPPM, data[1]);
    }
    __syncthreads();*/
    
#if SINGLETHREADMODE
    if (i == 0 && j == 0) {
    for(i = 0; i < Nx; i++)
    {
        for(j = 0; j < Ny; j++)
#else
    while (i < Nx)
    {
        j = threadIdx.y + blockDim.y * blockIdx.y;
        
        while (j < Ny)
#endif
        {
            /* Do all of this within the while loop to prevent threads with i's
             * and j's outside of the image boundaris from accessing rayDoubles.
             */
            double dx = data[2] / (double) Nx;
            double dy = data[3] / (double) Ny;

            double ttrueFinal = 0.0;
            int finalObj = 0;
            bool hitObject = false;
            
            int rayInd = (j * Nx + i) * 26;
            double *finalNewA = &rayDoubles[rayInd];
            double *finalNewB = &rayDoubles[rayInd + 3];
            double *pointA = &rayDoubles[rayInd + 6];
            double *newA = &rayDoubles[rayInd + 9];
            double *newB = &rayDoubles[rayInd + 12];
            double *coeffs = &rayDoubles[rayInd + 15];
            double *intersect = &rayDoubles[rayInd + 18];
            double *intersectNormal = &rayDoubles[rayInd + 21];
            double *roots = &rayDoubles[rayInd + 24];
            
            double *lDoubles = &lightDoubles[(j * Nx + i) * 32];
        
            // The positions are subtracted by a Nx/2 or Ny/2 term to center
            // the film plane
            double px = (i * dx) - (data[2] / (double) 2);
            double py = (j * dy) - (data[3] / (double) 2);
            double pxColor[] = {bgColor[0], bgColor[1], bgColor[2]};
            
            if (!antiAliased)
            {
                // Transform point A into film coordinates
                pointA[0] = (data[5] * e3[0]) + (px * e1[0]) + (py * e2[0]);
                pointA[1] = (data[5] * e3[1]) + (px * e1[1]) + (py * e2[1]);
                pointA[2] = (data[5] * e3[2]) + (px * e1[2]) + (py * e2[2]);
                
                hitObject = false;
                finalObj = 0, ttrueFinal = 0;
                for (int k = 0; k < data[0]; k++)
                {
                    // Find the ray equation transformations
                    newa(objects[k].unScale, objects[k].unRotate, pointA, newA);
                    newb(objects[k].unScale, objects[k].unRotate, 
                         objects[k].unTranslate, lookFrom, newB);

                    // Find the quadratic equation coefficients
                    findCoeffs(newA, newB, coeffs, true);
                    // Using the coefficients, find the roots
                    findRoots(coeffs, roots);

                    // Check to see if the roots are FLT_MAX - if they are then the 
                    // ray missed the superquadric. If they haven't missed then we 
                    // can continue with the calculations.
                    if (roots[0] != FLT_MAX)
                    {
                        // Use the update rule to find tfinal
                        double tini = min(roots[0], roots[1]);
                        double tfinal = updateRule(newA, newB, &objects[k].e, 
                                                   &objects[k].n, tini, data[4]);

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
                                finalNewA[0] = newA[0];
                                finalNewA[1] = newA[1];
                                finalNewA[2] = newA[2];
                                finalNewB[0] = newB[0];
                                finalNewB[1] = newB[1];
                                finalNewB[2] = newB[2];
                            }
                            else if (!hitObject)
                            {
                                hitObject = true;
                                ttrueFinal = tfinal;
                                finalObj = k;
                                finalNewA[0] = newA[0];
                                finalNewA[1] = newA[1];
                                finalNewA[2] = newA[2];
                                finalNewB[0] = newB[0];
                                finalNewB[1] = newB[1];
                                finalNewB[2] = newB[2];
                            }
                        }
                    }
                }
                if(hitObject)
                {
                    findRay(pointA, lookFrom, intersect, ttrueFinal);
                    unitNormal(objects[finalObj].rotate, finalNewA, finalNewB, 
                               intersectNormal, ttrueFinal, objects[finalObj].e, 
                               objects[finalObj].n);

                    lighting(intersect, intersectNormal, lookFrom,
                             &objects[finalObj].mat,
                             lightsPPM, data[1], objects, data[0], data[4],
                             finalObj, RECURSIONDEPTH,
                             &pxColor[0], lDoubles);
                }
            }
            else
            {
                double denom = 3 + (2 / sqrt((double) 2));
                double pxCoeffs[] = {(1 / (2 * sqrt((double) 2))) / denom,
                                     (1 / (double) 2) / denom,
                                     (1 / (2 * sqrt((double) 2))) / denom,
                                     (1 / (double) 2) / denom,
                                     1 / denom,
                                     (1 / (double) 2) / denom,
                                     (1 / (2 * sqrt((double) 2))) / denom,
                                     (1 / (double) 2) / denom,
                                     (1 / (2 * sqrt((double) 2))) / denom};
                int counter = 0;
                for (int g = -1; g <= 1; g++)
                {
                    for (int h = -1; h <= 1; h++)
                    {
                        double thisPx = px + (g * (dx / (double) 2));
                        double thisPy = py + (h * (dy / (double) 2));
                        
                        // Transform point A into film Coordinates
                        pointA[0] = (data[5] * e3[0]) + (thisPx * e1[0]) + (thisPy * e2[0]);
                        pointA[1] = (data[5] * e3[1]) + (thisPx * e1[1]) + (thisPy * e2[1]);
                        pointA[2] = (data[5] * e3[2]) + (thisPx * e1[2]) + (thisPy * e2[2]);
                        
                        hitObject = false;
                        finalObj = 0, ttrueFinal = 0;
                        for (int k = 0; k < data[0]; k++)
                        {
                            // Find the ray equation transformations
                            newa(objects[k].unScale, objects[k].unRotate, 
                                 pointA, newA);
                            newb(objects[k].unScale, objects[k].unRotate, 
                                 objects[k].unTranslate, lookFrom, newB);

                            // Find the quadratic equation coefficients
                            findCoeffs(newA, newB, coeffs, true);
                            // Using the coefficients, find the roots
                            findRoots(coeffs, roots);

                            // Check to see if the roots are FLT_MAX - if they are then the 
                            // ray missed the superquadric. If they haven't missed then we 
                            // can continue with the calculations.
                            if (roots[0] != FLT_MAX)
                            {
                                // Use the update rule to find tfinal
                                double tini = min(roots[0], roots[1]);
                                double tfinal = updateRule(newA, newB, &objects[k].e, 
                                                           &objects[k].n, tini, data[4]);

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
                                        finalNewA[0] = newA[0];
                                        finalNewA[1] = newA[1];
                                        finalNewA[2] = newA[2];
                                        finalNewB[0] = newB[0];
                                        finalNewB[1] = newB[1];
                                        finalNewB[2] = newB[2];
                                    }
                                    else if (!hitObject)
                                    {
                                        hitObject = true;
                                        ttrueFinal = tfinal;
                                        finalObj = k;
                                        finalNewA[0] = newA[0];
                                        finalNewA[1] = newA[1];
                                        finalNewA[2] = newA[2];
                                        finalNewB[0] = newB[0];
                                        finalNewB[1] = newB[1];
                                        finalNewB[2] = newB[2];
                                    }
                                }
                            }
                        }
                        if(hitObject)
                        {
                            findRay(pointA, lookFrom, intersect, ttrueFinal);
                            unitNormal(objects[finalObj].rotate, finalNewA, 
                                       finalNewB, intersectNormal, ttrueFinal, 
                                       objects[finalObj].e, objects[finalObj].n);

                            double color[] = {0, 0, 0};
                            
                            lighting(intersect, intersectNormal, lookFrom,
                                     &objects[finalObj].mat,
                                     lightsPPM, data[1], objects, data[0], 
                                     data[4],
                                     finalObj, RECURSIONDEPTH, 
                                     &color[0], lDoubles);

                            pxColor[0] += color[0] * pxCoeffs[counter];
                            pxColor[1] += color[1] * pxCoeffs[counter];
                            pxColor[2] += color[2] * pxCoeffs[counter];
                        }
                        counter++;
                    }
                }
            }
            int index = (j * Nx + i) * 3;
            grid[index] = pxColor[0];
            grid[index + 1] = pxColor[1];
            grid[index + 2] = pxColor[2];
            
#if SINGLETHREADMODE
        }
    }
    }
}
#else
            j += blockDim.y * gridDim.y;
        }
        i += blockDim.x * gridDim.x;
    }
}
#endif

void callRaytraceKernel(double *grid, Object *objects, Point_Light *lightsPPM, 
                        double *data, double *bgColor, double *e1, double *e2, 
                        double *e3, double *lookFrom, int Nx, int Ny,
                        bool antiAliased, int blockPower) 
{
    int blockSize = pow(2, blockPower);
    
    dim3 blocks;
    blocks.x = blockSize;
    blocks.y = blockSize;
    
    int gx = (Nx / blockSize);
    int gy = (Ny / blockSize);
    if (gx < 1) gx = 1;
    if (gy < 1) gy = 1;
    dim3 gridSize;
    gridSize.x = gx;
    gridSize.y = gy;

    // Mostly debug info, but possibly interesting
    int numThreads = (blockSize * gx) * (blockSize * gy);
    printf("Image size: %d x %d (%d Pixels)\n", Nx, Ny, Nx * Ny);
    printf("Total number of threads: %d\n", (blockSize * gx) * (blockSize * gy));
    
    float factor = numThreads / (float) (1024 * 1024);
    size_t deviceLimit;
    gpuErrChk(cudaDeviceGetLimit(&deviceLimit, cudaLimitStackSize));
    printf("Original Device stack size: %d\n", (int) deviceLimit);
    printf("Total Device stack memory: %0.2f MB\n", (int) deviceLimit * factor);
    
    // Recursion's a bitch, gotta increase that stack size
    // (Also relevant for images larger than 400 x 400 or so, I suppose)
    gpuErrChk(cudaDeviceSetLimit(cudaLimitStackSize, 4096));
    gpuErrChk(cudaDeviceGetLimit(&deviceLimit, cudaLimitStackSize));
    printf("New Device stack size: %d\n", (int) deviceLimit);
    printf("Total Device stack memory: %0.2f MB\n", (int) deviceLimit * factor);

    // Allocate space on the gpu for the double arrays in the kernel
    double *rayDoubles;
    gpuErrChk(cudaMalloc(&rayDoubles, sizeof(double) * Nx * Ny * 26));
    gpuErrChk(cudaMemset(rayDoubles, 0, sizeof(double) * Nx * Ny * 26));
    
    double *lightDoubles;
    gpuErrChk(cudaMalloc(&lightDoubles, sizeof(double) * Nx * Ny * 32));
    gpuErrChk(cudaMemset(lightDoubles, 0, sizeof(double) * Nx * Ny * 32));
    
    raytraceKernel<<<gridSize, blocks>>>(grid, objects, lightsPPM, data, 
                                         bgColor, e1, e2, e3, lookFrom, 
                                         rayDoubles, lightDoubles, Nx, Ny, 
                                         antiAliased);
    gpuErrChk(cudaPeekAtLastError());
    gpuErrChk(cudaDeviceSynchronize());
    gpuErrChk(cudaFree(rayDoubles));
    gpuErrChk(cudaFree(lightDoubles));
}
