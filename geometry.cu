

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



__host__ __device__ float length(float ��_1_x, float ��_1_y, float ��_2_x, float ��_2_y)
{
	return sqrt(powf(��_1_x - ��_2_x, 2) + powf(��_1_y - ��_2_y, 2));
}

__host__ __device__ float length(point ��_1, point ��_2)
{
	return length(��_1[0], ��_1[1], ��_2[0], ��_2[1]);
}

__host__ __device__ point rotate(const point ԭ��, const point ��_2, float �Ƕ�, bool rad)
{
	if (!rad)
	{
		�Ƕ� = deg2rad(�Ƕ�);
	}
	float x = (��_2[0] - ԭ��[0]) * cos(�Ƕ�) - (��_2[1] - ԭ��[1]) * sin(�Ƕ�) + ԭ��[0];
	float y = (��_2[0] - ԭ��[0]) * sin(�Ƕ�) + (��_2[1] - ԭ��[1]) * cos(�Ƕ�) + ԭ��[1];
	return point(x, y);
}
