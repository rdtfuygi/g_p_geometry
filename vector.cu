#include "geometry.cuh"
#define _USE_MATH_DEFINES 
#include <math.h>

__host__ __device__ vector::vector() :point(1, 0) {}

__host__ __device__ vector::vector(float x, float y) : point(x, y) {}

__host__ __device__ vector::vector(point ��) : point(��) {};

__host__ __device__ vector::vector(float ����[2], float ����)
{
	float ���� = ���� / ::length({ 0,0 }, ����);
	locat[0] = ����[0] * ����;
	locat[1] = ����[1] * ����;
}

__host__ __device__ vector::vector(float �Ƕ�, bool rad, float ����)
{
	if (!rad)
	{
		�Ƕ� = deg2rad(�Ƕ�);
	}
	locat[0] = cos(�Ƕ�) * ����;
	locat[1] = sin(�Ƕ�) * ����;
}

__host__ __device__ vector& vector::operator+=(vector ����)
{
	locat[0] += ����[0];
	locat[1] += ����[1];
	return *this;
}

__host__ __device__ vector& vector::operator-=(vector ����)
{
	locat[0] -= ����[0];
	locat[1] -= ����[1];
	return *this;
}

__host__ __device__ vector& vector::operator*=(float ��)
{
	locat[0] *= ��;
	locat[1] *= ��;
	return *this;
}

__host__ __device__ vector& vector::operator/=(float ��)
{
	locat[0] /= ��;
	locat[1] /= ��;
	return *this;
}

__host__ __device__ vector vector::unitize() const
{
	float ���� = length();
	if (���� < 1e-16)
	{
		return vector(float(M_SQRT1_2), float(M_SQRT1_2));
	}
	return vector(*this / ����);
}

__host__ __device__ float vector::length() const
{
	return ::length(*this);
}

__host__ __device__ vector vector::rotate(float �Ƕ�, bool rad) const
{
	return vector(::rotate({ 0,0 }, point(*this), �Ƕ�, rad));
}

__host__ __device__ float vector::angle_get(bool rad) const
{
	float �Ƕ� = atan(locat[1] / locat[0]) + (locat[0] > 0 ? 0 : M_PI);
	if (!rad)
	{
		�Ƕ� = rad2deg(�Ƕ�);
	}
	return �Ƕ�;
}

void vector::print(cv::InputOutputArray ͼ��, float ����, const cv::Scalar& ��ɫ, int ��ϸ) const
{
	int �� = ͼ��.rows(), �� = ͼ��.cols();
	int ԭ��_x = �� / 2, ԭ��_y = �� / 2;

	//int �Ŵ� = 2 * (�� > �� ? �� : ��);

	cv::Point ��_1(ԭ��_x, ԭ��_y);
	cv::Point ��_2(locat[0] * ���� + ԭ��_x, -locat[1] * ����  + ԭ��_y);
	cv::line(ͼ��, ��_1, ��_2, ��ɫ, ��ϸ);
}

__host__ __device__ float inc_angle_cos(vector ����_1, vector ����_2)
{	
	return ����_1.unitize() * ����_2.unitize();
}

__host__ __device__ float inc_angle_sin(vector ����_1, vector ����_2)
{
	return ����_1.unitize() ^ ����_2.unitize();
}

__host__ __device__ vector operator-(vector ����)
{
	return vector(-����[0], -����[1]);
}

__host__ __device__ vector operator+(vector ����_1, vector ����_2)
{
	return vector(����_1[0] + ����_2[0], ����_1[1] + ����_2[1]);
}

__host__ __device__ vector operator-(vector ����_1, vector ����_2)
{
	return vector(����_1[0] - ����_2[0], ����_1[1] - ����_2[1]);
}

__host__ __device__ vector operator*(vector ����, float ��)
{
	return vector(����[0] * ��, ����[1] * ��);
}

__host__ __device__ vector operator*(float ��, vector ����)
{
	return vector(����[0] * ��, ����[1] * ��);
}

__host__ __device__ vector operator/(vector ����, float ��)
{
	return vector(����[0] / ��, ����[1] / ��);
}


__host__ __device__ float length(vector ����)
{
	return length({ 0,0 }, ����);
}



__host__ __device__ float operator*(vector ����_1, vector ����_2)
{
	return ����_1[0] * ����_2[0] + ����_1[1] * ����_2[1];
}

__host__ __device__ float operator^(vector ����_1, vector ����_2)
{
	return (����_1[0] * ����_2[1]) - (����_1[1] * ����_2[0]);
}