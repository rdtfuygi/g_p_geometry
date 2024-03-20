#pragma once


#ifdef GPGEOMETRY_EXPORTS
#define DLL __declspec(dllexport)
#else
#define DLL __declspec(dllimport)//导入
#endif


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <vector>
#include <opencv.hpp>


DLL __host__ __device__ float deg2rad(float rad);
DLL __host__ __device__ float rad2deg(float deg);


class DLL point
{
public:
	float locat[2];
	__host__ __device__ point();
	__host__ __device__ point(float x, float y);
	__host__ __device__ point(float 位置[2]);
	__host__ __device__ float& operator[](int i);
	__host__ __device__ float operator[](int i) const;
	void print(cv::InputOutputArray 图像, float 比例, const cv::Scalar& 颜色, int 粗细 = 1) const;
};

DLL __host__ __device__ float length(float 点_1_x, float 点_1_y, float 点_2_x, float 点_2_y);
DLL __host__ __device__ float length(point 点_1, point 点_2);

DLL __host__ __device__ point rotate(const point 原点, const point 点_2, float 角度, bool rad = false);

class DLL vector :public point
{
public:
	__host__ __device__ vector();
	__host__ __device__ vector(float x, float y);
	__host__ __device__ vector(point 点);
	__host__ __device__ vector(float 方向[2], float 长度);
	__host__ __device__ vector(float 角度, bool rad = false, float 长度 = 1);
	__host__ __device__ vector& operator += (vector 向量);
	__host__ __device__ vector& operator -= (vector 向量);
	__host__ __device__ vector& operator *= (float 数);
	__host__ __device__ vector& operator /= (float 数);
	__host__ __device__ vector unitize() const;
	__host__ __device__ float length() const;
	__host__ __device__ vector rotate(float 角度, bool rad = false) const;

	__host__ __device__ float angle_get(bool rad = false) const;

	void print(cv::InputOutputArray 图像, float 比例, const cv::Scalar& 颜色, int 粗细 = 1) const;
};

DLL __host__ __device__ float inc_angle_cos(vector 向量_1, vector 向量_2);
DLL __host__ __device__ float inc_angle_sin(vector 向量_1, vector 向量_2);

DLL __host__ __device__ vector operator - (vector 向量);

DLL __host__ __device__ vector operator + (vector 向量_1, vector 向量_2);
DLL __host__ __device__ vector operator - (vector 向量_1, vector 向量_2);
DLL __host__ __device__ vector operator * (vector 向量, float 数);
DLL __host__ __device__ vector operator * (float 数, vector 向量);
DLL __host__ __device__ vector operator / (vector 向量, float 数);

DLL __host__ __device__ float operator * (vector 向量_1, vector 向量_2);
DLL __host__ __device__ float operator ^ (vector 向量_1, vector 向量_2);

DLL __host__ __device__ float length(vector 向量);



class DLL  line
{
public:
	point origin;
	vector dir;
	__host__ __device__ line();
	__host__ __device__ line(point 点, vector 向量);
	__host__ __device__ line(point 点, float 角度, bool rad = false);
	__host__ __device__ line(float 点_1_x, float 点_1_y, float 点_2_x, float 点_2_y);
	__host__ __device__ line(point 点_1, point 点_2);
	__host__ __device__ line(float k, float b);
	__host__ __device__ point point_get(float t) const;
	__host__ __device__ float angle_get(bool rad = false) const;
	__host__ __device__ line rotate(const point 点, float 角度, bool rad = false) const;
	__host__ __device__ float point_dist(const point 点) const;
	__host__ __device__ void norm();

	void print(cv::InputOutputArray 图像, float 比例, const cv::Scalar& 颜色, int 粗细 = 1) const;
};

class DLL  ray :public line
{
public:
	__host__ __device__ ray();
	__host__ __device__ ray(point 点, vector 向量);
	__host__ __device__ ray(point 点, float 角度, bool rad = false);
	__host__ __device__ ray(float 点_1_x, float 点_1_y, float 点_2_x, float 点_2_y);
	__host__ __device__ ray(point 点_1, point 点_2);
	__host__ __device__ ray rotate(const point 点, float 角度, bool rad = false) const;
	__host__ __device__ float point_dist(const point 点) const;

	void print(cv::InputOutputArray 图像, float 比例, const cv::Scalar& 颜色, int 粗细 = 1) const;
};

class DLL  seg :public ray
{
public:
	float dist;
	__host__ __device__ seg();
	__host__ __device__ seg(point 点, vector 向量, float 长度);
	__host__ __device__ seg(point 点, float 方向, float 长度, bool rad = false);
	__host__ __device__ seg(float 点_1_x, float 点_1_y, float 点_2_x, float 点_2_y);
	__host__ __device__ seg(point 点_1, point 点_2);
	__host__ __device__ point end() const;
	__host__ __device__ seg rotate(const point 点, float 角度, bool rad = false) const;
	__host__ __device__ float point_dist(const point 点) const;
	__host__ __device__ void norm();

	void print(cv::InputOutputArray 图像, float 比例, const cv::Scalar& 颜色, int 粗细 = 1) const;
};

DLL __host__ __device__ float inc_angle_cos(const line l_1, const line l_2);
DLL __host__ __device__ float inc_angle_sin(const line l_1, const line l_2);
DLL __host__ __device__ void cross(const line l_1, const line l_2, float& t_1, float& t_2);
DLL __host__ __device__ void cross(const line l_1, const ray l_2, float& t_1, float& t_2);
DLL __host__ __device__ void cross(const line l_1, const seg l_2, float& t_1, float& t_2);
DLL __host__ __device__ void cross(const ray l_1, const line l_2, float& t_1, float& t_2);
DLL __host__ __device__ void cross(const ray l_1, const ray l_2, float& t_1, float& t_2);
DLL __host__ __device__ void cross(const ray l_1, const seg l_2, float& t_1, float& t_2);
DLL __host__ __device__ void cross(const seg l_1, const line l_2, float& t_1, float& t_2);
DLL __host__ __device__ void cross(const seg l_1, const ray l_2, float& t_1, float& t_2);
DLL __host__ __device__ void cross(const seg l_1, const seg l_2, float& t_1, float& t_2);
DLL __host__ __device__ point cross(const line l_1, const line l_2);
DLL __host__ __device__ point cross(const line l_1, const ray l_2);
DLL __host__ __device__ point cross(const line l_1, const seg l_2);
DLL __host__ __device__ point cross(const ray l_1, const line l_2);
DLL __host__ __device__ point cross(const ray l_1, const ray l_2);
DLL __host__ __device__ point cross(const ray l_1, const seg l_2);
DLL __host__ __device__ point cross(const seg l_1, const line l_2);
DLL __host__ __device__ point cross(const seg l_1, const ray l_2);
DLL __host__ __device__ point cross(const seg l_1, const seg l_2);
DLL __host__ __device__ bool is_cross(const line l_1, const line l_2);
DLL __host__ __device__ bool is_cross(const line l_1, const ray l_2);
DLL __host__ __device__ bool is_cross(const line l_1, const seg l_2);
DLL __host__ __device__ bool is_cross(const ray l_1, const line l_2);
DLL __host__ __device__ bool is_cross(const ray l_1, const ray l_2);
DLL __host__ __device__ bool is_cross(const ray l_1, const seg l_2);
DLL __host__ __device__ bool is_cross(const seg l_1, const line l_2);
DLL __host__ __device__ bool is_cross(const seg l_1, const ray l_2);
DLL __host__ __device__ bool is_cross(const seg l_1, const seg l_2);


class DLL tirangle
{
public:
	seg segs[3];

	__host__ __device__ tirangle();
	__host__ __device__ tirangle(const point* 点);
	tirangle(std::vector<point>& 点);

	__host__ __device__ seg& operator[](int i);
	__host__ __device__ seg operator[](int i) const;

	__host__ __device__ void reset_seg();
	__host__ __device__ void reset_seg(int i);

	__host__ __device__ bool is_cross(const seg l) const;

	__host__ __device__ float area() const;

	void print(cv::InputOutputArray 图像, float 比例, const cv::Scalar& 颜色, int 粗细 = 1) const;
};


class DLL poly
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
	seg segs[20];

	__host__ __device__ void changed() const;
public:

	__host__ __device__ poly();
	__host__ __device__ poly(const point* 点, int m = 20);
	poly(std::vector<point>& 点);
	__host__ __device__ poly(const tirangle 三角);

	__host__ __device__ bool legal() const;

	__host__ __device__ void point_get(point*& 点);
	void point_get(std::vector<point>& 点);
	__host__ __device__ void seg_get(seg*& 线段);
	void seg_get(std::vector<seg>& 线段);

	__host__ __device__ bool point_in(point 点) const;

	__host__ __device__ void reset_seg();

	__host__ __device__ void reset_seg(int i);

	__host__ __device__ bool is_overlap(const poly other) const;

	__host__ __device__ bool full_overlap(const poly other) const;

	__host__ __device__ float overlap_area(const poly other) const;

	__host__ __device__ seg& operator[](int i);

	__host__ __device__ seg operator[](int i) const;

	__host__ __device__ float dir_area() const;

	__host__ __device__ float area() const;

	void print(cv::InputOutputArray 图像, float 比例, const cv::Scalar& 颜色, int 粗细 = 1) const;

	__host__ __device__ point center() const;

	__host__ __device__ point fast_center() const;

	__host__ __device__ vector move2center();

	__host__ __device__ void simple(float 角度 = 30, bool rad = false);
};

DLL __host__ __device__ bool is_overlap(const poly p_1, const poly p_2);

DLL __host__ __device__ float overlap_area(const poly p_1, const poly p_2);

DLL __host__ __device__ float dist(const poly p_1, const poly p_2);

DLL __host__ __device__ float dist(const poly p, const line l);

DLL __host__ __device__ float dist(const poly p, const ray l);

DLL __host__ __device__ float dist(const poly p, const seg l);


