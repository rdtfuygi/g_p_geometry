#pragma once
#define _USE_MATH_DEFINES 


#ifdef GPGEOMETRY_EXPORTS
#define GPGEOMETRY_DLL __declspec(dllexport)
#else
#define GPGEOMETRY_DLL __declspec(dllimport)//����
#endif


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <vector>

//#define no_opencv

#ifndef no_opencv
#include <opencv.hpp>
#endif 




GPGEOMETRY_DLL __host__ __device__ float deg2rad(float rad);
GPGEOMETRY_DLL __host__ __device__ float rad2deg(float deg);


class GPGEOMETRY_DLL point
{
public:
	float locat[2];
	__host__ __device__ point();
	__host__ __device__ point(float x, float y);
	__host__ __device__ point(float λ��[2]);
	__host__ __device__ float& operator[](int i);
	__host__ __device__ float operator[](int i) const;

#ifndef no_opencv
	void print(cv::InputOutputArray ͼ��, float ����, const cv::Scalar& ��ɫ, int ��ϸ = 1) const;
#endif
};

GPGEOMETRY_DLL __host__ __device__ float length(float ��_1_x, float ��_1_y, float ��_2_x, float ��_2_y);
GPGEOMETRY_DLL __host__ __device__ float length(point ��_1, point ��_2);

GPGEOMETRY_DLL __host__ __device__ point rotate(const point ԭ��, const point ��_2, float �Ƕ�, bool rad = false);

class GPGEOMETRY_DLL vector :public point
{
public:
	__host__ __device__ vector();
	__host__ __device__ vector(float x, float y);
	__host__ __device__ vector(point ��);
	__host__ __device__ vector(float ����[2], float ����);
	__host__ __device__ vector(float �Ƕ�, bool rad = false, float ���� = 1);
	__host__ __device__ vector& operator += (vector ����);
	__host__ __device__ vector& operator -= (vector ����);
	__host__ __device__ vector& operator *= (float ��);
	__host__ __device__ vector& operator /= (float ��);
	__host__ __device__ vector unitize() const;
	__host__ __device__ float length() const;
	__host__ __device__ vector rotate(float �Ƕ�, bool rad = false) const;

	__host__ __device__ float angle_get(bool rad = false) const;

#ifndef no_opencv
	void print(cv::InputOutputArray ͼ��, float ����, const cv::Scalar& ��ɫ, int ��ϸ = 1) const;
#endif
};

GPGEOMETRY_DLL __host__ __device__ float inc_angle_cos(vector ����_1, vector ����_2);
GPGEOMETRY_DLL __host__ __device__ float inc_angle_sin(vector ����_1, vector ����_2);

GPGEOMETRY_DLL __host__ __device__ vector operator - (vector ����);

GPGEOMETRY_DLL __host__ __device__ vector operator + (vector ����_1, vector ����_2);
GPGEOMETRY_DLL __host__ __device__ vector operator - (vector ����_1, vector ����_2);
GPGEOMETRY_DLL __host__ __device__ vector operator * (vector ����, float ��);
GPGEOMETRY_DLL __host__ __device__ vector operator * (float ��, vector ����);
GPGEOMETRY_DLL __host__ __device__ vector operator / (vector ����, float ��);

GPGEOMETRY_DLL __host__ __device__ float operator * (vector ����_1, vector ����_2);
GPGEOMETRY_DLL __host__ __device__ float operator ^ (vector ����_1, vector ����_2);

GPGEOMETRY_DLL __host__ __device__ float length(vector ����);



class GPGEOMETRY_DLL  line
{
public:
	point origin;
	vector dir;
	__host__ __device__ line();
	__host__ __device__ line(point ��, vector ����);
	__host__ __device__ line(point ��, float �Ƕ�, bool rad = false);
	__host__ __device__ line(float ��_1_x, float ��_1_y, float ��_2_x, float ��_2_y);
	__host__ __device__ line(point ��_1, point ��_2);
	__host__ __device__ line(float k, float b);
	__host__ __device__ point point_get(float t) const;
	__host__ __device__ float angle_get(bool rad = false) const;
	__host__ __device__ line rotate(const point ��, float �Ƕ�, bool rad = false) const;
	__host__ __device__ float point_dist(const point ��) const;
	__host__ __device__ void norm();

#ifndef no_opencv
	void print(cv::InputOutputArray ͼ��, float ����, const cv::Scalar& ��ɫ, int ��ϸ = 1) const;
#endif
};

class GPGEOMETRY_DLL  ray :public line
{
public:
	__host__ __device__ ray();
	__host__ __device__ ray(point ��, vector ����);
	__host__ __device__ ray(point ��, float �Ƕ�, bool rad = false);
	__host__ __device__ ray(float ��_1_x, float ��_1_y, float ��_2_x, float ��_2_y);
	__host__ __device__ ray(point ��_1, point ��_2);
	__host__ __device__ ray rotate(const point ��, float �Ƕ�, bool rad = false) const;
	__host__ __device__ float point_dist(const point ��) const;
#ifndef no_opencv
	void print(cv::InputOutputArray ͼ��, float ����, const cv::Scalar& ��ɫ, int ��ϸ = 1) const;
#endif
};

class GPGEOMETRY_DLL  seg :public ray
{
public:
	float dist;
	__host__ __device__ seg();
	__host__ __device__ seg(point ��, vector ����, float ����);
	__host__ __device__ seg(point ��, float ����, float ����, bool rad = false);
	__host__ __device__ seg(float ��_1_x, float ��_1_y, float ��_2_x, float ��_2_y);
	__host__ __device__ seg(point ��_1, point ��_2);
	__host__ __device__ point end() const;
	__host__ __device__ seg rotate(const point ��, float �Ƕ�, bool rad = false) const;
	__host__ __device__ float point_dist(const point ��) const;
	__host__ __device__ void norm();

#ifndef no_opencv
	void print(cv::InputOutputArray ͼ��, float ����, const cv::Scalar& ��ɫ, int ��ϸ = 1) const;
#endif
};

GPGEOMETRY_DLL __host__ __device__ float inc_angle_cos(const line l_1, const line l_2);
GPGEOMETRY_DLL __host__ __device__ float inc_angle_sin(const line l_1, const line l_2);
GPGEOMETRY_DLL __host__ __device__ void cross(const line l_1, const line l_2, float& t_1, float& t_2);
GPGEOMETRY_DLL __host__ __device__ void cross(const line l_1, const ray l_2, float& t_1, float& t_2);
GPGEOMETRY_DLL __host__ __device__ void cross(const line l_1, const seg l_2, float& t_1, float& t_2);
GPGEOMETRY_DLL __host__ __device__ void cross(const ray l_1, const line l_2, float& t_1, float& t_2);
GPGEOMETRY_DLL __host__ __device__ void cross(const ray l_1, const ray l_2, float& t_1, float& t_2);
GPGEOMETRY_DLL __host__ __device__ void cross(const ray l_1, const seg l_2, float& t_1, float& t_2);
GPGEOMETRY_DLL __host__ __device__ void cross(const seg l_1, const line l_2, float& t_1, float& t_2);
GPGEOMETRY_DLL __host__ __device__ void cross(const seg l_1, const ray l_2, float& t_1, float& t_2);
GPGEOMETRY_DLL __host__ __device__ void cross(const seg l_1, const seg l_2, float& t_1, float& t_2);
GPGEOMETRY_DLL __host__ __device__ point cross(const line l_1, const line l_2);
GPGEOMETRY_DLL __host__ __device__ point cross(const line l_1, const ray l_2);
GPGEOMETRY_DLL __host__ __device__ point cross(const line l_1, const seg l_2);
GPGEOMETRY_DLL __host__ __device__ point cross(const ray l_1, const line l_2);
GPGEOMETRY_DLL __host__ __device__ point cross(const ray l_1, const ray l_2);
GPGEOMETRY_DLL __host__ __device__ point cross(const ray l_1, const seg l_2);
GPGEOMETRY_DLL __host__ __device__ point cross(const seg l_1, const line l_2);
GPGEOMETRY_DLL __host__ __device__ point cross(const seg l_1, const ray l_2);
GPGEOMETRY_DLL __host__ __device__ point cross(const seg l_1, const seg l_2);
GPGEOMETRY_DLL __host__ __device__ bool is_cross(const line l_1, const line l_2);
GPGEOMETRY_DLL __host__ __device__ bool is_cross(const line l_1, const ray l_2);
GPGEOMETRY_DLL __host__ __device__ bool is_cross(const line l_1, const seg l_2);
GPGEOMETRY_DLL __host__ __device__ bool is_cross(const ray l_1, const line l_2);
GPGEOMETRY_DLL __host__ __device__ bool is_cross(const ray l_1, const ray l_2);
GPGEOMETRY_DLL __host__ __device__ bool is_cross(const ray l_1, const seg l_2);
GPGEOMETRY_DLL __host__ __device__ bool is_cross(const seg l_1, const line l_2);
GPGEOMETRY_DLL __host__ __device__ bool is_cross(const seg l_1, const ray l_2);
GPGEOMETRY_DLL __host__ __device__ bool is_cross(const seg l_1, const seg l_2);


class GPGEOMETRY_DLL tirangle
{
public:
	seg segs[3];

	__host__ __device__ tirangle();
	__host__ __device__ tirangle(const point* ��);
	tirangle(std::vector<point>& ��);

	__host__ __device__ seg& operator[](int i);
	__host__ __device__ seg operator[](int i) const;

	__host__ __device__ void reset_seg();
	__host__ __device__ void reset_seg(int i);

	__host__ __device__ bool is_cross(const seg l) const;

	__host__ __device__ float area() const;

#ifndef no_opencv
	void print(cv::InputOutputArray ͼ��, float ����, const cv::Scalar& ��ɫ, int ��ϸ = 1) const;
#endif
};


class GPGEOMETRY_DLL poly
{
protected:
	mutable bool legal_change;
	mutable bool area_change;
	mutable bool dir_area_change;
	mutable bool center_change;
	mutable bool fast_center_change;

	mutable bool legal_;
	mutable float area_;
	mutable float dir_area_;
	mutable point center_;
	mutable point fast_center_;

	mutable bool seg_change;
	seg segs[16];

	__host__ __device__ void changed() const;
public:

	__host__ __device__ poly();
	__host__ __device__ poly(const point* ��, int m = 16);
	poly(std::vector<point>& ��);
	__host__ __device__ poly(const tirangle ����);

	__host__ __device__ bool legal() const;

	__host__ __device__ void point_get(point*& ��);
	void point_get(std::vector<point>& ��);
	__host__ __device__ void seg_get(seg*& �߶�);
	void seg_get(std::vector<seg>& �߶�);

	__host__ __device__ bool point_in(point ��) const;

	__host__ __device__ void reset_seg();

	__host__ __device__ void reset_seg(int i);

	__host__ __device__ bool is_overlap(const poly other) const;

	__host__ __device__ bool full_overlap(const poly other) const;

	__host__ __device__ float overlap_area(const poly other) const;

	__host__ __device__ seg& operator[](int i);

	__host__ __device__ seg operator[](int i) const;

	__host__ __device__ float dir_area() const;

	__host__ __device__ float area() const;

#ifndef no_opencv
	void print(cv::InputOutputArray ͼ��, float ����, const cv::Scalar& ��ɫ, int ��ϸ = 1) const;
#endif

	__host__ __device__ point center() const;

	__host__ __device__ point fast_center() const;

	__host__ __device__ vector move2center();

	__host__ __device__ void simple(float �Ƕ� = 30, bool rad = false);
};

GPGEOMETRY_DLL __host__ __device__ bool is_overlap(const poly p_1, const poly p_2);

GPGEOMETRY_DLL __host__ __device__ float overlap_area(const poly p_1, const poly p_2);

GPGEOMETRY_DLL __host__ __device__ float dist(const poly p_1, const poly p_2);

GPGEOMETRY_DLL __host__ __device__ float dist(const poly p, const line l);

GPGEOMETRY_DLL __host__ __device__ float dist(const poly p, const ray l);

GPGEOMETRY_DLL __host__ __device__ float dist(const poly p, const seg l);


