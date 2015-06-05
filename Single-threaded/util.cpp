/* CS/CNS 171 Fall 2014
 *
 * This file contains the actual implementations of the functions in util.h.
 */

#include "util.h"
#include <math.h>
#include <float.h>

/* Gets the rotation matrix for a given rotation axis (x, y, z) and angle. */
MatrixXd get_rotate_mat(double x, double y, double z, double angle)
{
    double nor = sqrt((x * x) + (y * y) + (z * z));
    x = x / nor;
    y = y / nor;
    z = z / nor;
    angle = deg2rad(angle);

    MatrixXd rotate = MatrixXd::Identity(4,4);
    rotate(0,0) = pow(x,2) + (1 - pow(x,2)) * cos(angle);
    rotate(0,1) = (x * y * (1 - cos(angle))) - (z * sin(angle));
    rotate(0,2) = (x * z * (1 - cos(angle))) + (y * sin(angle));

    rotate(1,0) = (y * x * (1 - cos(angle))) + (z * sin(angle));
    rotate(1,1) = pow(y,2) + (1 - pow(y,2)) * cos(angle);
    rotate(1,2) = (y * z * (1 - cos(angle))) - (x * sin(angle));

    rotate(2,0) = (z * x * (1 - cos(angle))) - (y * sin(angle));
    rotate(2,1) = (z * y * (1 - cos(angle))) + (x * sin(angle));
    rotate(2,2) = pow(z,2) + (1 - pow(z,2)) * cos(angle);

    return rotate;
}

/* Gets the scaling matrix for a given scaling vector (x, y, z). */
MatrixXd get_scale_mat(double x, double y, double z)
{
    MatrixXd scale = MatrixXd::Identity(4,4);
    scale(0,0) = x;
    scale(1,1) = y;
    scale(2,2) = z;
    return scale;
}

/* Takes a 3x3 MatrixXd and turns it into a 4x4 MatrixXd. */
MatrixXd matrix3to4(MatrixXd mat)
{
    MatrixXd mat2 = MatrixXd::Identity(4,4);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            mat2(i,j) = mat(i,j);
        }
    }

    return mat2;
}

/* Takes a 4x4 MatrixXd and turns it into a 3x3 MatrixXd. */
MatrixXd matrix4to3(MatrixXd mat)
{
    MatrixXd mat2 = MatrixXd::Identity(3,3);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            mat2(i,j) = mat(i,j);
        }
    }

    double w = mat(3,3);
    if (w != 0.0)
    {
        mat2 = mat2 / (double) w;
    }

    return mat2;
}

/* Implicit Superquadric function. */
double isq(Vector3d vec, double e, double n)
{
    // Test for n = 0 now to prevent divide-by-zero errors.
    if (n == 0)
        return FLT_MAX;
    
    double zTerm = pow(pow(vec(2), 2.0), 1.0 / (double) n);

    // Test for e = 0 now to prevent divide-by-zero errors.
    if (e == 0)
        return zTerm;

    double xTerm = pow(pow(vec(0), 2.0), 1.0 / (double) e);
    double yTerm = pow(pow(vec(1), 2.0), 1.0 / (double) e);
    double xyTerm = pow(xTerm + yTerm, e / (double) n);
    return xyTerm + zTerm - 1.0;
}

/* Ray Equation */
Vector3d findRay(Vector3d a, Vector3d b, double t)
{
    return (a * t) + b;
}

/* Apply the Inverse Transform to a to get a new, usable a. */
Vector3d newa(MatrixXd unScale, MatrixXd unRotate, Vector3d a)
{
    return unScale * (unRotate * a);
}

/* Apply the Inverse Transform to b to get a new, usable b. */
Vector3d newb(MatrixXd unScale, MatrixXd unRotate, Vector3d unTranslate, Vector3d b)
{
    return unScale * (unRotate * (b + unTranslate));
}

/* Finds the scalar coefficients of the quadratic equation with the two given
 * vectors. If positiveb is true then the returned coeffs will all be multiplied
 * by -1 if be is negative, to ensure that b is positive. */
Vector3d findCoeffs(Vector3d a, Vector3d b, bool positiveb)
{
    Vector3d coeffs(a.dot(a), 2 * a.dot(b), b.dot(b) - 3);
    if (positiveb)
    {
        if (coeffs(1) < 0)
            coeffs = coeffs * -1;
    }
    return coeffs;
}

/* Finds the roots of the quadratic with the coefficients specified by the input
 * Vector3d. If one of the roots is complex then FLT_MAX is returned instead. */
Vector2d findRoots(Vector3d coeffs)
{
    double tMinus = 0.0, tPlus = 0.0;
    double interior = pow(coeffs(1), 2) - (4 * coeffs(0) * coeffs(2));
    if (interior < 0)
    {
        tMinus = tPlus = FLT_MAX;
    }
    else
    {
        tMinus = (-coeffs(1) - sqrt(interior)) / (double) (2 * coeffs(0));
        tPlus = (2 * coeffs(2)) / (double) (-coeffs(1) - sqrt(interior));
    }
    Vector2d roots(tMinus, tPlus);
    return roots;
}

/* Uses Newton's method to find the t value at which a ray hits the superquadric.
 * If the ray actually misses the superquadric then FLT_MAX is returned instead.*/
double updateRule(Vector3d a, Vector3d b, double e, double n, double t, double epsilon)
{
    Vector3d vec = findRay(a, b, t);
    double gP = gPrime(vec, a, e, n);
    double gPPrevious = gP;
    double g = 0.0;
    double tnew = t, told = t;
    bool stopPoint = false;

    while (!stopPoint)
    {
        told = tnew;
        vec = findRay(a, b, told);
        gP = gPrime(vec, a, e, n);
        g = isq(vec, e, n);

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

/* Returns -1 for negative numbers, 1 for positive numbers, and 0 for zero. */
int sign(double s)
{
    if(s > 0) return 1;
    if(s < 0) return -1;
    return 0;
}

/* Gradient of the isq function. */
Vector3d isqGradient(Vector3d vec, double e, double n)
{
    double xval = 0.0, yval = 0.0, zval = 0.0;
    // Check for n = 0 to prevent divide-by-zero errors
    if (n == 0)
    {
        cout << "n is zero!" << endl;
        xval = yval = zval = FLT_MAX;
    }
    // Check for e = 0 to prevent divide-by-zero errors
    else if (e == 0)
    {
        cout << "e is  zero!" << endl;
        xval = yval = FLT_MAX;
        zval = (2 * vec(2) * pow(pow(vec(2), 2), ((double) 1 / n) - 1)) / (double) n;
    }
    else
    {
        double xterm = pow(pow(vec(0), 2.0), (double) 1 / e);
        double yterm = pow(pow(vec(1), 2.0), (double) 1 / e);
        double xyterm = pow(xterm + yterm, ((double) e / n) - 1);
        double x2term = (2 * vec(0) * pow(pow(vec(0), 2.0), ((double) 1 / e) - 1));
        double y2term = (2 * vec(1) * pow(pow(vec(1), 2.0), ((double) 1 / e) - 1));
        xval = x2term * xyterm / (double) n;
        yval = y2term * xyterm / (double) n;
        zval = (2 * vec(2) * pow(pow(vec(2), 2.0), ((double) 1 / n) - 1)) / (double) n;
    }
    
    Vector3d res(xval, yval, zval);
    return res;
}

/* Derivative of the isq function. */
double gPrime(Vector3d vec, Vector3d a, double e, double n)
{
    return a.dot(isqGradient(vec, e, n));
}

/* Unit normal vector at a point on the superquadric */
Vector3d unitNormal(MatrixXd r, Vector3d vec1, Vector3d vec2, double tt, double e, double n)
{
    Vector3d nor = isqGradient(findRay(vec1, vec2, tt), e, n);
    nor = r * nor;
    nor.normalize();
    return nor;
}

// Simple function to convert an angle in degrees to radians
double deg2rad(double angle)
{
    return angle * M_PI / 180.0;
}
// Simple function to convert an angle in radians to degrees
double rad2deg(double angle)
{
    return angle * (180.0 / M_PI);
}

double vectorAngle(Vector3d a, Vector3d b)
{
    double dot = a.dot(b);
    double mag = a.norm() * b.norm();

    return acos(dot / mag);
}

// Calculates the refracted ray from an input ray and normal and a snell ratio
// If there is total internal reflection, then a vector of FLT_MAX is returned
// instead.
Vector3d refractedRay(Vector3d a, Vector3d n, double snell)
{
    double cos1 = (-1 * n).dot(a);
    if (cos1 < 0)
    {
        n = -1 * n;
        cos1 = (-1 * n).dot(a);
    }
    double radicand = 1 - (pow(snell, 2) * (1 - pow(cos1,2)));

    if (radicand < 0)
    {
        Vector3d res(FLT_MAX, FLT_MAX, FLT_MAX);
        return res;
    }
    else
    {
        double cos2 = sqrt(radicand);

        return (snell * a) + (((snell * cos1) - cos2) * n);
    }
}
