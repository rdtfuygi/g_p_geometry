#include "geometry.cuh"

__host__ __device__ line::line() :origin(), dir() {}

__host__ __device__ line::line(point ��, vector ����) : origin(��), dir(����.unitize()) {}

__host__ __device__ line::line(point ��, float �Ƕ�, bool rad) : origin(��)
{
	if (!rad)
	{
		�Ƕ� = deg2rad(�Ƕ�);
	}
	dir[0] = cos(�Ƕ�);
	dir[1] = sin(�Ƕ�);
}

__host__ __device__ line::line(float ��_1_x, float ��_1_y, float ��_2_x, float ��_2_y) :origin(��_1_x, ��_1_y), dir((vector(��_2_x, ��_2_y) - vector(��_1_x, ��_1_y)).unitize()) {}

__host__ __device__ line::line(point ��_1, point ��_2) :origin(��_1), dir((vector(��_2) - vector(��_1)).unitize()) {}

__host__ __device__ line::line(float k, float b) :origin(0, b), dir(vector(1, k + b).unitize()) {}

__host__ __device__ point line::point_get(float t) const
{
	return point(vector(origin) + (dir * t));
}

__host__ __device__ float line::angle_get(bool rad) const
{
	return dir.angle_get(rad);
}

__host__ __device__ line line::rotate(const point ��, float �Ƕ�, bool rad) const
{
	return line(::rotate(��, origin, �Ƕ�, rad), dir.rotate(�Ƕ�, rad));
}

__host__ __device__ float line::point_dist(const point ��) const
{
	line temp;
	temp.origin = ��;
	temp.dir[0] = dir[1];
	temp.dir[1] = -dir[0];

	float t_1, t_2;
	cross(*this, temp, t_1, t_2);
	return abs(t_2);
}

__host__ __device__ void line::norm()
{
	float l = length(dir);
	dir /= l;
}

#ifndef no_opencv
void line::print(cv::InputOutputArray ͼ��, float ����, const cv::Scalar& ��ɫ, int ��ϸ) const
{
	int �� = ͼ��.rows(), �� = ͼ��.cols();
	int ԭ��_x = �� / 2, ԭ��_y = �� / 2;

	int �Ŵ� = 2 * (�� > �� ? �� : ��);

	cv::Point ��_1(origin[0] * ���� - dir[0] * �Ŵ� + ԭ��_x, -origin[1] * ���� + dir[1] * �Ŵ� + ԭ��_y);
	cv::Point ��_2(origin[0] * ���� + dir[0] * �Ŵ� + ԭ��_x, -origin[1] * ���� - dir[1] * �Ŵ� + ԭ��_y);
	cv::line(ͼ��, ��_1, ��_2, ��ɫ, ��ϸ);
}
#endif

__host__ __device__ float inc_angle_cos(const line l_1, const line l_2)
{
	return l_1.dir * l_2.dir;
}

__host__ __device__ float inc_angle_sin(const line l_1, const line l_2)
{
	return l_1.dir ^ l_2.dir;
}

__host__ __device__ void cross(const line l_1, const line l_2, float& t_1, float& t_2)
{
	float temp = l_1.dir ^ l_2.dir;
	if (temp == 0.0f)
	{
		t_1 = FLT_MAX;
		t_2 = FLT_MAX;
		return;
	}
	vector oo = vector(l_2.origin) - vector(l_1.origin);
	t_1 = (oo ^ l_2.dir) / temp;
	t_2 = (oo ^ l_1.dir) / temp;
}

__host__ __device__ void cross(const line l_1, const ray l_2, float& t_1, float& t_2)
{
	cross(line(l_1), line(l_2), t_1, t_2);
	if (0 > t_2)
	{
		t_1 = FLT_MAX;
		t_2 = FLT_MAX;
	}
}

__host__ __device__ void cross(const line l_1, const seg l_2, float& t_1, float& t_2)
{
	cross(line(l_1), line(l_2), t_1, t_2);
	if ((0 > t_2) || (t_2 > l_2.dist))
	{
		t_1 = FLT_MAX;
		t_2 = FLT_MAX;
	}
}

__host__ __device__ point cross(const line l_1, const line l_2)
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

__host__ __device__ point cross(const line l_1, const ray l_2)
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

__host__ __device__ point cross(const line l_1, const seg l_2)
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

__host__ __device__ bool is_cross(const line l_1, const line l_2)
{	
	float t_1, t_2;
	cross(l_1, l_2, t_1, t_2);
	if (t_1 == FLT_MAX)
	{
		return false;
	}
	return true;
}

__host__ __device__ bool is_cross(const line l_1, const ray l_2)
{
	float t_1, t_2;
	cross(l_1, l_2, t_1, t_2);
	if (t_1 == FLT_MAX)
	{
		return false;
	}
	return true;
}

__host__ __device__ bool is_cross(const line l_1, const seg l_2)
{
	float t_1, t_2;
	cross(l_1, l_2, t_1, t_2);
	if (t_1 == FLT_MAX)
	{
		return false;
	}
	return true;
}
