#include "geometry.cuh"

__host__ __device__ ray::ray() :line() {}

__host__ __device__ ray::ray(point ��, vector ����) :line(��, ����) {}

__host__ __device__ ray::ray(point ��, float �Ƕ�, bool rad) :line(��, �Ƕ�, rad) {}

__host__ __device__ ray::ray(float ��_1_x, float ��_1_y, float ��_2_x, float ��_2_y) :line(��_1_x, ��_1_y, ��_2_x, ��_2_y) {}

__host__ __device__ ray::ray(point ��_1, point ��_2) :line(��_1, ��_2) {}

__host__ __device__ ray ray::rotate(const point ��, float �Ƕ�, bool rad) const
{
	return ray(::rotate(��, origin, �Ƕ�, rad), dir.rotate(�Ƕ�, rad));
}

__host__ __device__ float ray::point_dist(const point ��) const
{
	line temp;
	temp.origin = ��;
	temp.dir[0] = dir[1];
	temp.dir[1] = -dir[0];

	float t_1, t_2;
	cross(line(*this), temp, t_1, t_2);
	if (t_1 > 0)
	{
		return abs(t_2);
	}
	else
	{
		return length(��, origin);
	}
}

#ifndef no_opencv
void ray::print(cv::InputOutputArray ͼ��, float ����, const cv::Scalar& ��ɫ, int ��ϸ) const
{
	int �� = ͼ��.rows(), �� = ͼ��.cols();
	int ԭ��_x = �� / 2, ԭ��_y = �� / 2;

	int �Ŵ� = 2 * (�� > �� ? �� : ��);

	cv::Point ��_1(origin[0] * ���� + ԭ��_x, -origin[1] * ���� + ԭ��_y);
	cv::Point ��_2(origin[0] * ���� + dir[0] * �Ŵ� + ԭ��_x, -origin[1] * ���� - dir[1] * �Ŵ� + ԭ��_y);
	cv::line(ͼ��, ��_1, ��_2, ��ɫ, ��ϸ);
}
#endif

__host__ __device__ void cross(const ray l_1, const line l_2, float& t_1, float& t_2)
{
	cross(line(l_1), line(l_2), t_1, t_2);
	if (0 > t_1)
	{
		t_1 = FLT_MAX;
		t_2 = FLT_MAX;
	}
}

__host__ __device__ void cross(const ray l_1, const ray l_2, float& t_1, float& t_2)
{
	cross(line(l_1), line(l_2), t_1, t_2);
	if ((0 > t_1) || (0 > t_2))
	{
		t_1 = FLT_MAX;
		t_2 = FLT_MAX;
	}
}

__host__ __device__ void cross(const ray l_1, const seg l_2, float& t_1, float& t_2)
{
	point end_1 = point((l_1.dir[0] > 0 ? FLT_MAX : -FLT_MAX), (l_1.dir[1] > 0 ? FLT_MAX : -FLT_MAX));
	point end_2 = l_2.end();
	if ((fmin(l_1.origin[0], end_1[0]) > fmax(l_2.origin[0], end_2[0])) ||
		(fmin(l_1.origin[1], end_1[1]) > fmax(l_2.origin[1], end_2[1])) ||
		(fmin(l_2.origin[0], end_2[0]) > fmax(l_1.origin[0], end_1[0])) ||
		(fmin(l_2.origin[1], end_2[1]) > fmax(l_1.origin[1], end_1[1])))
	{
		t_2 = FLT_MAX;
		t_1 = FLT_MAX;
		return;
	}
	cross(line(l_1), line(l_2), t_1, t_2);
	if ((0 > t_1) || (0 > t_2) || (t_2 > l_2.dist))
	{
		t_1 = FLT_MAX;
		t_2 = FLT_MAX;
	}
}

__host__ __device__ point cross(const ray l_1, const line l_2)
{
	float t_1, t_2;
	cross(l_1, l_2, t_1, t_2);
	if (t_1 != FLT_MAX)
	{
		return l_1.point_get(t_1);
	}
	else
	{
		return point(FLT_MAX, FLT_MAX);
	}
}

__host__ __device__ point cross(const ray l_1, const ray l_2)
{
	float t_1, t_2;
	cross(l_1, l_2, t_1, t_2);
	if (t_1 != FLT_MAX)
	{
		return l_1.point_get(t_1);
	}
	else
	{
		return point(FLT_MAX, FLT_MAX);
	}
}

__host__ __device__ point cross(const ray l_1, const seg l_2)
{
	float t_1, t_2;
	cross(l_1, l_2, t_1, t_2);
	if (t_1 != FLT_MAX)
	{
		return l_1.point_get(t_1);
	}
	else
	{
		return point(FLT_MAX, FLT_MAX);
	}
}

__host__ __device__ bool is_cross(const ray l_1, const seg l_2)
{
	float t_1, t_2;
	cross(l_1, l_2, t_1, t_2);
	if (t_1 == FLT_MAX)
	{
		return false;
	}
	return true;
}

__host__ __device__ bool is_cross(const ray l_1, const line l_2)
{
	float t_1, t_2;
	cross(l_1, l_2, t_1, t_2);
	if (t_1 == FLT_MAX)
	{
		return false;
	}
	return true;
}

__host__ __device__ bool is_cross(const ray l_1, const ray l_2)
{
	float t_1, t_2;
	cross(l_1, l_2, t_1, t_2);
	if (t_1 == FLT_MAX)
	{
		return false;
	}
	return true;
}

