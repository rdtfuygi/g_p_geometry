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

__host__ __device__ point::point(float λ��[2])
{
	locat[0] = λ��[0];
	locat[1] = λ��[1];
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
void point::print(cv::InputOutputArray ͼ��, float ����, const cv::Scalar& ��ɫ, int ��ϸ) const
{
	int �� = ͼ��.rows(), �� = ͼ��.cols();
	int ԭ��_x = �� / 2, ԭ��_y = �� / 2;

	//int �Ŵ� = 2 * (�� > �� ? �� : ��);

	cv::Point ��(locat[0] * ���� + ԭ��_x, -locat[1] * ���� + ԭ��_y);

	cv::circle(ͼ��, ��, ��ϸ, ��ɫ, -1);
}
#endif

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