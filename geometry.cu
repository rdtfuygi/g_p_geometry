#include "geometry.cuh"

__host__ __device__ float deg2rad(float rad)
{
	return rad * float(M_PI) / 180;
}

__host__ __device__ float rad2deg(float deg)
{
	return deg / float(M_PI) * 180;
}




