
#include <cstdio>
#include <math.h>

#include <cuda_runtime.h>

struct Point_Light;
struct Material;
struct Object;

void callRaytraceKernel(double *grid, Object *objects, Point_Light *lightsPPM, 
                        double *data, double *bgColor, double *e1, double *e2, 
                        double *e3, double *lookFrom, int Nx, int Ny,
                        bool antiAliased, int blockPower);
