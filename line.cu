#include "geometry.cuh"

__host__ __device__ line::line() :origin(), dir() {}

__host__ __device__ line::line(point 点, vector 向量) : origin(点), dir(向量.unitize()) {}

__host__ __device__ line::line(point 点, float 角度, bool rad) : origin(点)
{
	if (!rad)
	{
		角度 = deg2rad(角度);
	}
	dir[0] = cos(角度);
	dir[1] = sin(角度);
}

__host__ __device__ line::line(float 点_1_x, float 点_1_y, float 点_2_x, float 点_2_y) :origin(点_1_x, 点_1_y), dir((vector(点_2_x, 点_2_y) - vector(点_1_x, 点_1_y)).unitize()) {}

__host__ __device__ line::line(point 点_1, point 点_2) :origin(点_1), dir((vector(点_2) - vector(点_1)).unitize()) {}

__host__ __device__ line::line(float k, float b) :origin(0, b), dir(vector(1, k + b).unitize()) {}

__host__ __device__ point line::point_get(float t) const
{
	return point(vector(origin) + (dir * t));
}

__host__ __device__ float line::angle_get(bool rad) const
{
	return dir.angle_get(rad);
}

__host__ __device__ line line::rotate(const point 点, float 角度, bool rad) const
{
	return line(::rotate(点, origin, 角度, rad), dir.rotate(角度, rad));
}

__host__ __device__ float line::point_dist(const point 点) const
{
	line temp;
	temp.origin = 点;
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
void line::print(cv::InputOutputArray 图像, float 比例, const cv::Scalar& 颜色, int 粗细) const
{
	int 高 = 图像.rows(), 宽 = 图像.cols();
	int 原点_x = 宽 / 2, 原点_y = 高 / 2;

	int 放大 = 2 * (高 > 宽 ? 高 : 宽);

	cv::Point 点_1(origin[0] * 比例 - dir[0] * 放大 + 原点_x, -origin[1] * 比例 + dir[1] * 放大 + 原点_y);
	cv::Point 点_2(origin[0] * 比例 + dir[0] * 放大 + 原点_x, -origin[1] * 比例 - dir[1] * 放大 + 原点_y);
	cv::line(图像, 点_1, 点_2, 颜色, 粗细);
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
