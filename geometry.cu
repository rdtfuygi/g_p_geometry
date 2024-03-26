

#include "geometry.cuh"
#include <limits>


__host__ __device__ float deg2rad(float rad)
{
	return rad * float(M_PI) / 180;
}

__host__ __device__ float rad2deg(float deg)
{
	return deg / float(M_PI) * 180;
}



__host__ __device__ float length(float 点_1_x, float 点_1_y, float 点_2_x, float 点_2_y)
{
	return sqrt(powf(点_1_x - 点_2_x, 2) + powf(点_1_y - 点_2_y, 2));
}

__host__ __device__ float length(point 点_1, point 点_2)
{
	return length(点_1[0], 点_1[1], 点_2[0], 点_2[1]);
}

__host__ __device__ point rotate(const point 原点, const point 点_2, float 角度, bool rad)
{
	if (!rad)
	{
		角度 = deg2rad(角度);
	}
	float x = (点_2[0] - 原点[0]) * cos(角度) - (点_2[1] - 原点[1]) * sin(角度) + 原点[0];
	float y = (点_2[0] - 原点[0]) * sin(角度) + (点_2[1] - 原点[1]) * cos(角度) + 原点[1];
	return point(x, y);
}
