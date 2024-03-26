#include "geometry.cuh"
#define _USE_MATH_DEFINES 
#include <math.h>


tirangle::tirangle()
{
	segs[0].origin = point(0.0f, 0.0f);
	segs[1].origin = point(1.0f, 0.0f);
	segs[2].origin = point(0.0f, 1.0f);
	segs[0].dir = vector(1.0f, 0.0f);
	segs[0].dir = vector(-float(M_SQRT1_2), float(M_SQRT1_2));
	segs[0].dir = vector(0.0f, -1.0f);
}

tirangle::tirangle(const point* ��)
{
	segs[0] = seg(��[0], ��[1]);
	segs[1] = seg(��[1], ��[2]);
	segs[2] = seg(��[2], ��[0]);
}

tirangle::tirangle(std::vector<point>& ��)
{
	segs[0] = seg(��[0], ��[1]);
	segs[1] = seg(��[1], ��[2]);
	segs[2] = seg(��[2], ��[0]);
}

__host__ __device__ seg& tirangle::operator[](int i)
{
	return segs[i];
}

__host__ __device__ seg tirangle::operator[](int i) const
{
	return segs[i];
}


__host__ __device__ void tirangle::reset_seg()
{
	for (int i = 0; i < 3; i++)
	{
		reset_seg(i);
	}
}

__host__ __device__ void tirangle::reset_seg(int i)
{
	segs[i] = seg(segs[i].origin, segs[(i + 1) % 3].origin);
}

__host__ __device__ bool tirangle::is_cross(const seg l) const
{
	return (::is_cross(segs[0], l)) || (::is_cross(segs[1], l)) || (::is_cross(segs[2], l));
}

__host__ __device__ float tirangle::area() const
{
	return abs((vector(segs[1].origin) - vector(segs[0].origin)) ^ (vector(segs[2].origin) - vector(segs[0].origin))) / 2;
}
#ifndef no_opencv
void tirangle::print(cv::InputOutputArray ͼ��, float ����, const cv::Scalar& ��ɫ, int ��ϸ) const
{
	seg(segs[0].origin, segs[1].origin).print(ͼ��, ����, ��ɫ, ��ϸ);
	seg(segs[1].origin, segs[2].origin).print(ͼ��, ����, ��ɫ, ��ϸ);
	seg(segs[2].origin, segs[0].origin).print(ͼ��, ����, ��ɫ, ��ϸ);
}
#endif