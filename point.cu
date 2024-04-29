#include "geometry.cuh"

__host__ __device__ point::point()
{
	locat[0] = 0;
	locat[1] = 0;
}

__host__ __device__ point::point(float x, float y)
{
	locat[0] = x;
	locat[1] = y;
}

__host__ __device__ point::point(float 位置[2])
{
	locat[0] = 位置[0];
	locat[1] = 位置[1];
}

__host__ __device__ float& point::operator[](int i)
{
	return locat[i & 1];
}

__host__ __device__ float point::operator[](int i) const
{
	return locat[i & 1];
}
#ifndef no_opencv
void point::print(cv::InputOutputArray 图像, float 比例, const cv::Scalar& 颜色, int 粗细) const
{
	int 高 = 图像.rows(), 宽 = 图像.cols();
	int 原点_x = 宽 / 2, 原点_y = 高 / 2;

	//int 放大 = 2 * (高 > 宽 ? 高 : 宽);

	cv::Point 点(locat[0] * 比例 + 原点_x, -locat[1] * 比例 + 原点_y);

	cv::circle(图像, 点, 粗细, 颜色, -1);
}
#endif

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