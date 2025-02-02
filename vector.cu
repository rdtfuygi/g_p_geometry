#include "geometry.cuh"
#define _USE_MATH_DEFINES 
#include <math.h>

__host__ __device__ vector::vector() :point(1, 0) {}

__host__ __device__ vector::vector(float x, float y) : point(x, y) {}

__host__ __device__ vector::vector(point 点) : point(点) {};

__host__ __device__ vector::vector(float 方向[2], float 长度)
{
	float 比例 = 长度 / ::length({ 0,0 }, 方向);
	locat[0] = 方向[0] * 比例;
	locat[1] = 方向[1] * 比例;
}

__host__ __device__ vector::vector(float 角度, bool rad, float 长度)
{
	if (!rad)
	{
		角度 = deg2rad(角度);
	}
	locat[0] = cos(角度) * 长度;
	locat[1] = sin(角度) * 长度;
}

__host__ __device__ vector& vector::operator+=(vector 向量)
{
	locat[0] += 向量[0];
	locat[1] += 向量[1];
	return *this;
}

__host__ __device__ vector& vector::operator-=(vector 向量)
{
	locat[0] -= 向量[0];
	locat[1] -= 向量[1];
	return *this;
}

__host__ __device__ vector& vector::operator*=(float 数)
{
	locat[0] *= 数;
	locat[1] *= 数;
	return *this;
}

__host__ __device__ vector& vector::operator/=(float 数)
{
	locat[0] /= 数;
	locat[1] /= 数;
	return *this;
}

__host__ __device__ vector vector::unitize() const
{
	float 长度 = length();
	if (长度 < 1e-16)
	{
		return vector(float(M_SQRT1_2), float(M_SQRT1_2));
	}
	return vector(*this / 长度);
}

__host__ __device__ float vector::length() const
{
	return ::length(*this);
}

__host__ __device__ vector vector::rotate(float 角度, bool rad) const
{
	return vector(::rotate({ 0,0 }, point(*this), 角度, rad));
}

__host__ __device__ float vector::angle_get(bool rad) const
{
	float 角度 = atan(locat[1] / locat[0]) + (locat[0] > 0 ? 0 : float(M_PI));
	if (!rad)
	{
		角度 = rad2deg(角度);
	}
	return 角度;
}

#ifndef no_opencv
void vector::print(cv::InputOutputArray 图像, float 比例, const cv::Scalar& 颜色, int 粗细) const
{
	int 高 = 图像.rows(), 宽 = 图像.cols();
	int 原点_x = 宽 / 2, 原点_y = 高 / 2;

	//int 放大 = 2 * (高 > 宽 ? 高 : 宽);

	cv::Point 点_1(原点_x, 原点_y);
	cv::Point 点_2(locat[0] * 比例 + 原点_x, -locat[1] * 比例  + 原点_y);
	cv::line(图像, 点_1, 点_2, 颜色, 粗细);
}
#endif

__host__ __device__ float inc_angle_cos(vector 向量_1, vector 向量_2)
{	
	return 向量_1.unitize() * 向量_2.unitize();
}

__host__ __device__ float inc_angle_sin(vector 向量_1, vector 向量_2)
{
	return 向量_1.unitize() ^ 向量_2.unitize();
}

__host__ __device__ vector operator-(vector 向量)
{
	return vector(-向量[0], -向量[1]);
}

__host__ __device__ vector operator+(vector 向量_1, vector 向量_2)
{
	return vector(向量_1[0] + 向量_2[0], 向量_1[1] + 向量_2[1]);
}

__host__ __device__ vector operator-(vector 向量_1, vector 向量_2)
{
	return vector(向量_1[0] - 向量_2[0], 向量_1[1] - 向量_2[1]);
}

__host__ __device__ vector operator*(vector 向量, float 数)
{
	return vector(向量[0] * 数, 向量[1] * 数);
}

__host__ __device__ vector operator*(float 数, vector 向量)
{
	return vector(向量[0] * 数, 向量[1] * 数);
}

__host__ __device__ vector operator/(vector 向量, float 数)
{
	return vector(向量[0] / 数, 向量[1] / 数);
}


__host__ __device__ float length(vector 向量)
{
	return length({ 0,0 }, 向量);
}



__host__ __device__ float operator*(vector 向量_1, vector 向量_2)
{
	return 向量_1[0] * 向量_2[0] + 向量_1[1] * 向量_2[1];
}

__host__ __device__ float operator^(vector 向量_1, vector 向量_2)
{
	return (向量_1[0] * 向量_2[1]) - (向量_1[1] * 向量_2[0]);
}