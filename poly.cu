#include "geometry.cuh"



__host__ __device__ void sort(double* list, int n, bool up = true)
{
	for (int i = n - 1; i > 0; i--)
	{
		bool swap = false;
		for (int j = 0; j < i; j++)
		{
			if ((list[j] > list[j + 1]) && up)
			{
				double temp_dist = list[j];
				list[j] = list[j + 1];
				list[j + 1] = temp_dist;
				swap = true;
			}
			else if ((list[j] < list[j + 1]) && !up)
			{
				double temp_dist = list[j];
				list[j] = list[j + 1];
				list[j + 1] = temp_dist;
				swap = true;
			}
		}
		if (!swap)
		{
			break;
		}
	}
}



__host__ __device__ poly::poly() {}

__host__ __device__ poly::poly(const point* 点, int m)
{
	int temp = m < 20 ? m : 20 ;
	for (int i = 0; i < temp - 1; i++)
	{
		segs[i] = seg(点[i], 点[i + 1]);
	}
	for (int i = temp; i < 20 ; i++)
	{
		segs[i] = seg(点[temp], 点[temp + 1]);
	}
	segs[20 - 1] = seg(点[temp - 1], 点[0]);
}

poly::poly(std::vector<point>& 点)
{
	int temp = (点.size() < 20 ) ? 点.size() : 20 ;
	for (int i = 0; i < temp - 1; i++)
	{
		segs[i] = seg(点[i], 点[i + 1]);
	}
	for (int i = temp - 1; i < 20 - 1; i++)
	{
		segs[i] = seg(点[点.size() - 1], 点[点.size() - 1]);
	}
	segs[20 - 1] = seg(点[temp - 1], 点[0]);
}

poly::poly(const tirangle 三角)
{
	segs[0] = 三角.segs[0];
	segs[1] = 三角.segs[1];
	segs[2] = 三角.segs[2];
	reset_seg();
}

__host__ __device__ bool poly::legal()
{
	reset_seg();
	for (int i = 1; i < 20; i++)
	{
		for (int j = 0; j < i - 1; j++)
		{
			double t_1, t_2;
			cross(segs[i], segs[j], t_1, t_2);
			if ((t_1 != DBL_MAX) && (((abs(t_1 - segs[i].dist) > 0.0001) && (abs(t_1) > 0.0001)) || ((abs(t_2 - segs[j].dist) > 0.0001) && (abs(t_2) > 0.0001))))
			{
				return false;
			}
		}
	}
	return true;
}

//__host__ __device__ double poly::one_link_area()
//{
//	point max = segs[0].origin, min = segs[0].origin;
//	for (int i = 1; i < 20; i++)
//	{
//		max[0] = (max[0] > segs[i].origin[0]) ? max[0] : segs[i].origin[0];
//		max[1] = (max[1] > segs[i].origin[1]) ? max[1] : segs[i].origin[1];
//		min[0] = (min[0] < segs[i].origin[0]) ? min[0] : segs[i].origin[0];
//		min[1] = (min[1] < segs[i].origin[1]) ? min[1] : segs[i].origin[1];
//	}
//
//
//	double last[20];
//	{
//		ray temp;
//		temp.origin = point(int(min[0] + 1), min[1]);
//		temp.dir = vector(0.0, 1.0);
//		for (int i = 0; i < 20; i++)
//		{
//			double t_1, t_2;
//			cross(temp, segs[i], t_1, t_2);
//			last[i] = t_1;
//		}
//		for (int i = 19; i > 0; i--)
//		{
//			bool swap = false;
//			for (int j = 0; j < i; j++)
//			{
//				if (last[j] < last[j + 1])
//				{
//					continue;
//				}
//				double temp_dist = last[j];
//				last[j] = last[j + 1];
//				last[j + 1] = temp_dist;
//				swap = true;
//			}
//			if (!swap)
//			{
//				break;
//			}
//		}
//	}
//
//
//	double areas[10];
//	char map[10] = { 0,1,2,3,4,5,6,7,8,9 };
//	for (int i = 0; i < 10; i++)
//	{
//		if ((last[2 * i + 1] != DBL_MAX) && (last[2 * i] != DBL_MAX))
//		{
//			areas[i] = last[2 * i + 1] - last[2 * i];
//		}
//		else
//		{
//			areas[i] = 0;
//		}
//	}
//
//	for (int x = min[0] + 2; x < max[0]; x++)
//	{
//		double dist[20];
//		char map_new[10] = { 10,10,10,10,10,10,10,10,10,10 };
//
//		seg temp;
//		temp.origin = point(x, min[1]);
//		temp.dir = vector(0.0, 1.0);
//		temp.dist = max[1] - min[1];
//		for (int i = 0; i < 20; i++)
//		{
//			double t_1, t_2;
//			cross(temp, segs[i], t_1, t_2);
//			dist[i] = t_1;
//		}
//		for (int i = 19; i > 0; i--)
//		{
//			bool swap = false;
//			for (int j = 0; j < i; j++)
//			{
//				if (dist[j] > dist[j + 1])
//				{
//					double temp_dist = dist[j];
//					dist[j] = dist[j + 1];
//					dist[j + 1] = temp_dist;
//					swap = true;
//				}
//			}
//			if (!swap)
//			{
//				break;
//			}
//		}
//
//		int i = 0, j = 0;
//		while ((i < 10) && (j < 10))
//		{
//			if ((last[2 * i] == DBL_MAX) || (last[2 * i + 1] == DBL_MAX) || (dist[2 * j] == DBL_MAX) || (dist[2 * j + 1] == DBL_MAX))
//			{
//				break;
//			}
//			if ((last[2 * i + 1] > dist[2 * j]) && (last[2 * i] < dist[2 * j + 1]))
//			{
//				if (map_new[j] == 10)
//				{
//					map_new[j] = map[i];
//					areas[map_new[j]] += dist[2 * j + 1] - dist[2 * j];
//				}
//				else if (map_new[j] != map[i])
//				{
//					areas[map_new[j]] += areas[map[i]];
//				}
//
//			}
//			if (last[2 * i + 1] < dist[2 * j + 1])
//			{
//				i++;
//			}
//			else if (last[2 * i + 1] > dist[2 * j + 1])
//			{
//				j++;
//			}
//			else
//			{
//				i++;
//				j++;
//			}
//		}
//		for (int i = 0; i < 10; i++)
//		{
//			last[2 * i] = dist[2 * i];
//			last[2 * i + 1] = dist[2 * i + 1];
//			if (map_new[i] != 10)
//			{
//				map[i] = map_new[i];
//			}
//		}
//	}
//	double output = 0;
//	for (int i = 0; i < 10; i++)
//	{
//		output = (areas[i] > output) ? areas[i] : output;
//	}
//
//	return output;
//}

__host__ __device__ void poly::point_get(point*& 点) const
{
	if (点 != nullptr)
	{
		delete[]点;
	}
	点 = new point[20];
	for (int i = 0; i < 20 ; i++)
	{
		点[i] = (segs[i]).origin;
	}
}

void poly::point_get(std::vector<point>& 点) const
{
	点 = std::vector<point>(20);
	for (int i = 0; i < 20 ; i++)
	{
		点[i] = (segs[i]).origin;
	}
}

__host__ __device__ void poly::seg_get(seg*& 线段) const
{
	if (线段 != nullptr)
	{
		delete[]线段;
	}
	线段 = new seg[20];
	for (int i = 0; i < 20 ; i++)
	{
		线段[i] = (segs[i]);
	}
}

void poly::seg_get(std::vector<seg>& 线段) const
{
	线段 = std::vector<seg>(20);
	for (int i = 0; i < 20 ; i++)
	{
		线段[i] = (segs[i]);
	}
}

__host__ __device__ bool poly::point_in(point 点) const
{
	ray temp;
	temp.origin = 点;
	temp.dir = vector(point({ 0,1 }));
	int k = 0;

	point max = segs[0].origin, min = segs[0].origin;
	for (int i = 1; i < 20 ; i++)
	{
		max[0] = (max[0] > segs[i].origin[0]) ? max[0] : segs[i].origin[0];
		max[1] = (max[1] > segs[i].origin[1]) ? max[1] : segs[i].origin[1];
		min[0] = (min[0] < segs[i].origin[0]) ? min[0] : segs[i].origin[0];
		min[1] = (min[1] < segs[i].origin[1]) ? min[1] : segs[i].origin[1];
	}
	if ((max[0] < 点[0]) || (max[1] < 点[1]) || (min[0] > 点[0]) || (min[1] > 点[1]))
	{
		return false;
	}

	for (int i = 0; i < 20 ; i++)
	{
		if (is_cross(temp, segs[i]))
		{
			k++;
		}
	}
	if ((k % 2) == 0)
	{
		return false;
	}

	temp.dir = vector(point({ 0,-1 }));
	k = 0;
	for (int i = 0; i < 20 ; i++)
	{
		if (is_cross(temp, segs[i]))
		{
			k++;
		}
	}
	if ((k % 2) == 0)
	{
		return false;
	}
	return true;
}

__host__ __device__ void poly::reset_seg()
{
	for (int i = 0, n = 0; (i < 20 - 1) && (n < 20); i++)
	{
		if ((abs(segs[i].origin[0] - segs[i + 1].origin[0]) > 0.001) || (abs(segs[i].origin[1] - segs[i + 1].origin[1]) > 0.001))
		{
			continue;
		}
		n++;
		i--;
		for (int j = i + 1; j < 20 - 1; j++)
		{
			segs[j].origin = segs[j + 1].origin;
		}
		segs[19].origin = segs[0].origin;
	}
	


	for (int i = 0; i < 20 - 1; i++)
	{
		segs[i] = seg(segs[i].origin, segs[i + 1].origin);
	}
	segs[19] = seg(segs[19].origin, segs[0].origin);
}

__host__ __device__ void poly::reset_seg(int i)
{
	segs[i] = seg(segs[i].origin, segs[(i + 1) % 20].origin);
}

__host__ __device__ seg& poly::operator[](int i)
{
	while (i < 0)
	{
		i += 20;
	}
	return segs[i % 20];
}

__host__ __device__ seg poly::operator[](int i) const
{
	while (i < 0)
	{
		i += 20;
	}
	return segs[i % 20];
}

__host__ __device__ double poly::dir_area() const
{
	double s = 0;
	for (int i = 0; i < 20 - 1; i++)
	{
		s += vector(segs[i].origin) ^ vector(segs[i + 1].origin);
	}
	s += vector(segs[20 - 1].origin) ^ vector(segs[0].origin);
	return s / 2;
}


__global__ void poly_area(seg* segs, double min_x, double min_y, double max_x, double max_y, double* output)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int x = min_x + idx;
	if (x >= max_x)
	{
		return;
	}
	double min[2] = { min_x,min_y };
	double max[2] = { max_x,max_y };

	seg temp;
	temp.origin = point(x, min[1]);
	temp.dir = vector(0.0, 1.0);
	temp.dist = max[1] - min[1];

	double dist[20];

	for (int i = 0; i < 20; i++)
	{
		double t_1, t_2;
		cross(temp, segs[i], t_1, t_2);
		dist[i] = t_1;
	}
	sort(dist, 20);
	//for (int i = 19; i > 0; i--)
	//{
	//	bool swap = false;
	//	for (int j = 0; j < i; j++)
	//	{
	//		if (dist[j] > dist[j + 1])
	//		{
	//			double temp_dist = dist[j];
	//			dist[j] = dist[j + 1];
	//			dist[j + 1] = temp_dist;
	//			swap = true;
	//		}
	//	}
	//	if (!swap)
	//	{
	//		break;
	//	}
	//}

	output[idx] = 0;
	for (int i = 0; i < 10; i++)
	{
		if ((dist[2 * i + 1] == DBL_MAX) || (dist[2 * i] == DBL_MAX))
		{
			break;
		}
		output[idx] += dist[2 * i + 1] - dist[2 * i];
	}
}

__host__ __device__ double poly::area() const
{
	point max = segs[0].origin, min = segs[0].origin;
	for (int i = 1; i < 20; i++)
	{
		max[0] = (max[0] > segs[i].origin[0]) ? max[0] : segs[i].origin[0];
		max[1] = (max[1] > segs[i].origin[1]) ? max[1] : segs[i].origin[1];
		min[0] = (min[0] < segs[i].origin[0]) ? min[0] : segs[i].origin[0];
		min[1] = (min[1] < segs[i].origin[1]) ? min[1] : segs[i].origin[1];
	}


#ifndef __CUDACC__
	int device_n;
	cudaGetDeviceCount(&device_n);

	if (((max[0] - min[0]) > 100) && (device_n > 0))
	{
		seg* segs_d = NULL;
		cudaMalloc((void**)&segs_d, sizeof(seg) * 20);
		cudaMemcpy(segs_d, segs, sizeof(seg) * 20, cudaMemcpyHostToDevice);
		double* output_d = NULL;
		cudaMalloc((void**)&output_d, sizeof(double) * int(max[0] - min[0]));

		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		int 每块线程 = deviceProp.maxThreadsPerBlock / 32;
		int 块 = int(max[0] - min[0]) / 每块线程 + 1;

		poly_area << < 块, 每块线程 >> > (segs_d, min[0], min[1], max[0], max[1], output_d);
		cudaFree(segs_d);

		double* output_h = new double[int(max[0] - min[0])];
		cudaMemcpy(output_h, output_d, sizeof(double) * int(max[0] - min[0]), cudaMemcpyDeviceToHost);
		cudaFree(output_d);
		double output = 0;
		for (int i = 0; i<int(max[0] - min[0]); i++)
		{
			output += output_h[i];
		}
		delete[]output_h;
		return output;
	}

#endif


	double output = 0;
	for (int x = min[0]; x < max[0]; x++)
	{
		seg temp;
		temp.origin = point(x, min[1]);
		temp.dir = vector(0.0, 1.0);
		temp.dist = max[1] - min[1];

		double dist[20];

		for (int i = 0; i < 20; i++)
		{
			double t_1, t_2;
			cross(temp, segs[i], t_1, t_2);
			dist[i] = t_1;
		}
		sort(dist, 20);
		//for (int i = 19; i > 0; i--)
		//{
		//	bool swap = false;
		//	for (int j = 0; j < i; j++)
		//	{
		//		if (dist[j] > dist[j + 1])
		//		{
		//			double temp_dist = dist[j];
		//			dist[j] = dist[j + 1];
		//			dist[j + 1] = temp_dist;
		//			swap = true;
		//		}
		//	}
		//	if (!swap)
		//	{
		//		break;
		//	}
		//}

		for (int i = 0; i < 10; i++)
		{
			if ((dist[2 * i + 1] == DBL_MAX) || (dist[2 * i] == DBL_MAX))
			{
				break;
			}
			output += dist[2 * i + 1] - dist[2 * i];
		}
	}
	return output;
}

void poly::print(cv::InputOutputArray 图像, double 比例, const cv::Scalar& 颜色, int 粗细) const
{
	//seg(segs[0].origin, segs[1].origin).print(图像, 比例, 颜色, 粗细 * 2);
	for (int i = 0; i < 19; i++)
	{
		seg(segs[i].origin, segs[i + 1].origin).print(图像, 比例, 颜色, 粗细);
	}
	seg(segs[19].origin, segs[0].origin).print(图像, 比例, 颜色, 粗细);
	//segs[0].origin.print(图像, 比例, 颜色, 粗细 * 4);
}

__global__ void poly_center(seg* segs, double min_x, double min_y, double max_x, double max_y, double* p_area, double* x_, double* y_)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int x = min_x + idx;
	if (x >= max_x)
	{
		return;
	}
	double min[2] = { min_x,min_y };
	double max[2] = { max_x,max_y };

	seg temp;
	temp.origin = point(x, min[1]);
	temp.dir = vector(0.0, 1.0);
	temp.dist = max[1] - min[1];

	double dist[20];

	for (int i = 0; i < 20; i++)
	{
		double t_1, t_2;
		cross(temp, segs[i], t_1, t_2);
		dist[i] = t_1;
	}
	sort(dist, 20);
	//for (int i = 19; i > 0; i--)
	//{
	//	bool swap = false;
	//	for (int j = 0; j < i; j++)
	//	{
	//		if (dist[j] > dist[j + 1])
	//		{
	//			double temp_dist = dist[j];
	//			dist[j] = dist[j + 1];
	//			dist[j + 1] = temp_dist;
	//			swap = true;
	//		}
	//	}
	//	if (!swap)
	//	{
	//		break;
	//	}
	//}

	p_area[idx] = 0, x_[idx] = 0, y_[idx] = 0;
	for (int i = 0; i < 10; i++)
	{
		if ((dist[2 * i + 1] == DBL_MAX) || (dist[2 * i] == DBL_MAX))
		{
			break;
		}
		p_area[idx] += dist[2 * i + 1] - dist[2 * i];
		y_[idx] += pow(dist[2 * i + 1] + min[1], 2) - pow(dist[2 * i] + min[1], 2);
	}
	x_[idx] = p_area[idx] * x;
}

__host__ __device__ point poly::center() const
{
	//double s = 0;
	//double x = 0, y = 0;
	//for (int i = 0; i < 19; i++)
	//{
	//	double 积 = segs[i].dir ^ segs[i + 1].dir;
	//	x += (segs[i].origin[0] + segs[i + 1].origin[0]) * 积;
	//	y += (segs[i].origin[1] + segs[i + 1].origin[1]) * 积;
	//	s += 积;
	//}
	//s *= 6;
	//return point(x / s, y / s);

	point max = segs[0].origin, min = segs[0].origin;
	for (int i = 1; i < 20; i++)
	{
		max[0] = (max[0] > segs[i].origin[0]) ? max[0] : segs[i].origin[0];
		max[1] = (max[1] > segs[i].origin[1]) ? max[1] : segs[i].origin[1];
		min[0] = (min[0] < segs[i].origin[0]) ? min[0] : segs[i].origin[0];
		min[1] = (min[1] < segs[i].origin[1]) ? min[1] : segs[i].origin[1];
	}


#ifndef __CUDACC__
	int device_n;
	cudaGetDeviceCount(&device_n);

	if (((max[0] - min[0]) > 100) && (device_n > 0))
	{
		seg* segs_d = NULL;//
		cudaMalloc((void**)&segs_d, sizeof(seg) * 20);
		cudaMemcpy(segs_d, segs, sizeof(seg) * 20, cudaMemcpyHostToDevice);
		double* p_area_d = NULL;//
		cudaMalloc((void**)&p_area_d, sizeof(double) * int(max[0] - min[0]));
		double* x_d = NULL;//
		cudaMalloc((void**)&x_d, sizeof(double) * int(max[0] - min[0]));
		double* y_d = NULL;//
		cudaMalloc((void**)&y_d, sizeof(double) * int(max[0] - min[0]));

		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		int 每块线程 = deviceProp.maxThreadsPerBlock / 32;
		int 块 = int(max[0] - min[0]) / 每块线程 + 1;

		poly_center << < 块, 每块线程 >> > (segs_d, min[0], min[1], max[0], max[1], p_area_d, x_d, y_d);
		cudaFree(segs_d);

		double* p_area_h = new double[int(max[0] - min[0])];
		cudaMemcpy(p_area_h, p_area_d, sizeof(double) * int(max[0] - min[0]), cudaMemcpyDeviceToHost);
		double* x_h = new double[int(max[0] - min[0])];
		cudaMemcpy(x_h, x_d, sizeof(double) * int(max[0] - min[0]), cudaMemcpyDeviceToHost);
		double* y_h = new double[int(max[0] - min[0])];
		cudaMemcpy(y_h, y_d, sizeof(double) * int(max[0] - min[0]), cudaMemcpyDeviceToHost);
		cudaFree(p_area_d);
		cudaFree(x_d);
		cudaFree(y_d);
		double p_area = 0, x = 0, y = 0;
		for (int i = 0; i<int(max[0] - min[0]); i++)
		{
			p_area += p_area_h[i];
			x += x_h[i];
			y += y_h[i];
		}
		delete[]p_area_h;
		delete[]x_h;
		delete[]y_h;
		return point(x / p_area, y / 2 / p_area);
	}

#endif


	double p_area = 0, x_ = 0, y_ = 0;
	for (int x = min[0]; x < max[0]; x++)
	{
		seg temp;
		temp.origin = point(x, min[1]);
		temp.dir = vector(0.0, 1.0);
		temp.dist = max[1] - min[1];

		double dist[20];

		for (int i = 0; i < 20; i++)
		{
			double t_1, t_2;
			cross(temp, segs[i], t_1, t_2);
			dist[i] = t_1;
		}
		sort(dist, 20);
		//for (int i = 19; i > 0; i--)
		//{
		//	bool swap = false;
		//	for (int j = 0; j < i; j++)
		//	{
		//		if (dist[j] > dist[j + 1])
		//		{
		//			double temp_dist = dist[j];
		//			dist[j] = dist[j + 1];
		//			dist[j + 1] = temp_dist;
		//			swap = true;
		//		}
		//	}
		//	if (!swap)
		//	{
		//		break;
		//	}
		//}

		double d_x = 0;
		for (int i = 0; i < 10; i++)
		{
			if ((dist[2 * i + 1] == DBL_MAX) || (dist[2 * i] == DBL_MAX))
			{
				break;
			}
			p_area += dist[2 * i + 1] - dist[2 * i];
			d_x += dist[2 * i + 1] - dist[2 * i];
			y_ += pow(dist[2 * i + 1] + min[1], 2) - pow(dist[2 * i] + min[1], 2);
		}
		x_ += d_x * x;
	}
	return point(x_ / p_area, y_ / 2 / p_area);
}

vector poly::move2center()
{
	vector move(center());

	for (int i = 0; i < 20; i++)
	{
		segs[i].origin = point(vector(segs[i].origin) - move);
	}

	return vector(0.0, 0.0) - move;
}

__host__ __device__ void poly::simple(double 角度, bool rad)
{
	if (!rad)
	{
		角度 = deg2rad(角度);
	}
	double cos_ = cos(角度);

	reset_seg();
	int n = 1;
	while (n != 0)	{

		n = 0;
		for (int i = 0, j = 1; j < 20; j++)
		{
			i = j - 1;

			double cos_t = (vector(0.0, 0.0) - segs[i].dir) * segs[j].dir;
			if ((cos_t < cos_) || (segs[i].dist < 0.0001) || (segs[j].dist < 0.0001))
			{
				continue;
			}

			n++;
			if (i == 18)
			{
				segs[19].origin = segs[0].origin;
			}

			for (int k = i + 1; k < 20 - 1; k++)
			{
				segs[k].origin = segs[k + 1].origin;
			}
			reset_seg();
		}

		vector dir_;
		for (int i = 19; i >= 0; i--)
		{
			if (segs[i].dist > 0.0001)
			{
				dir_ = segs[i].dir;
				break;
			}
		}

		double cos_t = (vector(0.0, 0.0) - dir_) * segs[0].dir;
		if (cos_t < cos_)
		{
			continue;
		}

		n++;
		for (int j = 0; j < 20 - 1; j++)
		{
			segs[j].origin = segs[j + 1].origin;
		}
		reset_seg();
	}
}

__host__ __device__ bool poly::is_overlap(const poly other) const
{
	return ::is_overlap(*this, other);
}

__host__ __device__ bool poly::full_overlap(const poly other) const
{
	for (int i = 0; i < 20; i++)
	{
		for (int j = 0; j < 20; j++)
		{
			if (is_cross(other[i], segs[j]))
			{
				return false;
			}
		}
		if (!point_in(other[i].origin))
		{
			return false;
		}
	}
	return true;
}

__host__ __device__ double poly::overlap_area(const poly other) const
{
	return ::overlap_area(*this, other);
}

__host__ __device__ bool is_overlap(const poly p_1, const poly p_2)
{
	int l[20];
	for (int i = 0; i < 20; i++)
	{
		l[i] = 0;
	}

	for (int i = 0; i < 20 ; i++)
	{
		int k = 0;
		for (int j = 0; j < 20; j++)
		{
			double t_1, t_2;
			cross(ray(p_1[i]), ray(p_2[j]), t_1, t_2);
			if ((t_1 < p_1[i].dist) && (t_2 < p_2[j].dist))
			{
				return true;
			}

			if ((t_1 != DBL_MAX) || ((t_2 > p_2[j].dist) && (t_2 != DBL_MAX)))
			{
				l[j]++;
				k++;
			}
		}

		if ((k % 2) == 0)
		{
			continue;
		}
		
		k = 0;
		for (int j = 0; j < 20; j++)
		{
			double t_1, t_2;
			cross(ray(p_1[i].origin, -1 * p_1[i].dir), p_2[j], t_1, t_2);
		
			if (t_1 != DBL_MAX)
			{
				k++;
			}
		}

		if ((k % 2) == 1)
		{
			return true;
		}
	}

	for (int i = 0; i < 20; i++)
	{
		if ((l[i] % 2) == 0)
		{
			continue;
		}
		
		l[i] = 0;
		for (int j = 0; j < 20 ; j++)
		{
			double t_1, t_2;
			cross(p_1[j], ray(p_2[i].origin, -1 * p_2[i].dir), t_1, t_2);
		
			if (t_1 != DBL_MAX)
			{
				l[i]++;
			}
		}

		if ((l[i] % 2) == 1)
		{
			return true;
		}
	}

	return false;
}


__global__ void overlap_area_cuda(poly* p, double min_x, double min_y, double max_x, double max_y, double* output)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int i = min_x + idx;
	if (i >= max_x)
	{
		return;
	}
	double min[2] = { min_x,min_y };
	double max[2] = { max_x,max_y };
	poly p_1 = p[0], p_2 = p[1];

	ray temp;
	temp.origin = point(i, min[1]);
	temp.dir = vector(0.0, 1.0);


	bool in_1 = false, in_2 = false;
	double dist[2][20];
	for (int j = 0; j < 20; j++)
	{
		double t_1, t_2;
		cross(temp, p_1.segs[j], t_1, t_2);
		dist[0][j] = t_1;
		if (t_1 != DBL_MAX)
		{
			in_1 = !in_1;
		}
		cross(temp, p_2.segs[j], t_1, t_2);
		dist[1][j] = t_1;
		if (t_1 != DBL_MAX)
		{
			in_2 = !in_2;
		}
	}
	sort(dist[0], 20);
	//for (int j = 19; j > 0; j--)
	//{
	//	bool swap = false;
	//	for (int k = 0; k < j; k++)
	//	{
	//		if (dist[0][k] > dist[0][k + 1])
	//		{
	//			swap = true;
	//			double t = dist[0][k];
	//			dist[0][k + 1] = dist[0][k];
	//			dist[0][k] = t;
	//		}
	//	}
	//	if (!swap)
	//	{
	//		break;
	//	}
	//}
	sort(dist[1], 20);
	//for (int j = 19; j > 0; j--)
	//{
	//	bool swap = false;
	//	for (int k = 0; k < j; k++)
	//	{
	//		if (dist[1][k] > dist[1][k + 1])
	//		{
	//			swap = true;
	//			double t = dist[1][k];
	//			dist[1][k + 1] = dist[1][k];
	//			dist[1][k] = t;
	//		}
	//	}
	//	if (!swap)
	//	{
	//		break;
	//	}
	//}

	int j = 0, k = 0;
	while ((j < 20) && (k < 20))
	{
		double next_1 = min[1] + dist[0][j] - temp.origin[1], next_2 = min[1] + dist[1][k] - temp.origin[1];

		if (in_1 && in_2 && ((next_1 < (max[1] - min[1])) || (next_2 < (max[1] - min[1]))))
		{
			output[idx] += fmin(next_1, next_2);
		}
		else if ((next_1 > (max[1] - min[1])) || (next_2 > (max[1] - min[1])))
		{
			break;
		}
		if (next_1 < next_2)
		{
			j++;
			in_1 = !in_1;
		}
		else if (next_1 > next_2)
		{
			k++;
			in_2 = !in_2;
		}
		else if(next_1 == next_2)
		{
			j++;
			k++;
			in_1 = !in_1;
			in_2 = !in_2;
		}

	}
}

__host__ __device__ double overlap_area(const poly p_1, const poly p_2)
{
	point max_1 = p_1.segs[0].origin, min_1 = p_1.segs[0].origin;
	for (int i = 1; i < 20; i++)
	{
		max_1[0] = (max_1[0] > p_1.segs[i].origin[0]) ? max_1[0] : p_1.segs[i].origin[0];
		max_1[1] = (max_1[1] > p_1.segs[i].origin[1]) ? max_1[1] : p_1.segs[i].origin[1];
		min_1[0] = (min_1[0] < p_1.segs[i].origin[0]) ? min_1[0] : p_1.segs[i].origin[0];
		min_1[1] = (min_1[1] < p_1.segs[i].origin[1]) ? min_1[1] : p_1.segs[i].origin[1];
	}
	point max_2 = p_2.segs[0].origin, min_2 = p_2.segs[0].origin;
	for (int i = 0; i < 20; i++)
	{
		max_2[0] = (max_2[0] > p_2.segs[i].origin[0]) ? max_2[0] : p_2.segs[i].origin[0];
		max_2[1] = (max_2[1] > p_2.segs[i].origin[1]) ? max_2[1] : p_2.segs[i].origin[1];
		min_2[0] = (min_2[0] < p_2.segs[i].origin[0]) ? min_2[0] : p_2.segs[i].origin[0];
		min_2[1] = (min_2[1] < p_2.segs[i].origin[1]) ? min_2[1] : p_2.segs[i].origin[1];
	}

	point max(fmin(max_1[0], max_2[0]), fmin(max_1[1], max_2[1])), min(fmax(min_1[0], min_2[0]), fmax(min_1[1], min_2[1]));


#ifndef __CUDACC__
	int device_n;
	cudaGetDeviceCount(&device_n);

	if (((max[0] - min[0]) > 100) && (device_n > 0))
	{
		poly p_h[2] = { p_1,p_2 };
		poly* p_d = NULL;
		cudaMalloc((void**)&p_d, sizeof(poly) * 2);
		cudaMemcpy(p_d, p_h, sizeof(poly) * 2, cudaMemcpyHostToDevice);

		double* output_d = NULL;
		cudaMalloc((void**)&output_d, sizeof(double) * int(max[0] - min[0]));

		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		int 每块线程 = deviceProp.maxThreadsPerBlock / 32;
		int 块 = int(max[0] - min[0]) / 每块线程 + 1;

		overlap_area_cuda << < 块, 每块线程 >> > (p_d, min[0], min[1], max[0], max[1], output_d);
		cudaFree(p_d);

		double* output_h = new double[int(max[0] - min[0])];
		cudaMemcpy(output_h, output_d, sizeof(double) * int(max[0] - min[0]), cudaMemcpyDeviceToHost);
		cudaFree(output_d);
		double output = 0;
		for (int i = 0; i<int(max[0] - min[0]); i++)
		{
			output += output_h[i];
		}
		delete[]output_h;
		return output;
	}

#endif

	double output = 0;
	for (int i = min[0]; i < max[0]; i++)
	{
		ray temp;
		temp.origin = point(i, min[1]);
		temp.dir = vector(0.0, 1.0);

		
		bool in_1 = false, in_2 = false;
		double dist[2][20];
		for (int j = 0; j < 20; j++)
		{
			double t_1, t_2;
			cross(temp, p_1.segs[j], t_1, t_2);
			dist[0][j] = t_1;
			if (t_1 != DBL_MAX)
			{
				in_1 = !in_1;
			}
			cross(temp, p_2.segs[j], t_1, t_2);
			dist[1][j] = t_1;
			if (t_1 != DBL_MAX)
			{
				in_2 = !in_2;
			}
		}
		sort(dist[0], 20);
		//for (int j = 19; j > 0; j--)
		//{
		//	bool swap = false;
		//	for (int k = 0; k < j; k++)
		//	{
		//		if (dist[0][k] > dist[0][k + 1])
		//		{
		//			swap = true;
		//			double t = dist[0][k];
		//			dist[0][k + 1] = dist[0][k];
		//			dist[0][k] = t;
		//		}
		//	}
		//	if (!swap)
		//	{
		//		break;
		//	}
		//}
		sort(dist[1], 20);
		//for (int j = 19; j > 0; j--)
		//{
		//	bool swap = false;
		//	for (int k = 0; k < j; k++)
		//	{
		//		if (dist[1][k] > dist[1][k + 1])
		//		{
		//			swap = true;
		//			double t = dist[1][k];
		//			dist[1][k + 1] = dist[1][k];
		//			dist[1][k] = t;
		//		}
		//	}
		//	if (!swap)
		//	{
		//		break;
		//	}
		//}

		int j = 0, k = 0;
		while ((j < 20) && (k < 20))
		{
			double next_1 = min[1] + dist[0][j] - temp.origin[1], next_2 = min[1] + dist[1][k] - temp.origin[1];

			if (in_1 && in_2 && ((next_1 < (max[1] - min[1])) || (next_2 < (max[1] - min[1]))))
			{
				output += fmin(next_1, next_2);
			}
			else if ((next_1 > (max[1] - min[1])) || (next_2 > (max[1] - min[1])))
			{
				break;
			}
			if (next_1 < next_2)
			{
				j++;
				in_1 = !in_1;
			}
			else if (next_1 > next_2)
			{
				k++;
				in_2 = !in_2;
			}
			else if (next_1 == next_2)
			{
				j++;
				k++;
				in_1 = !in_1;
				in_2 = !in_2;
			}
		}
	}
	return output;
}

__host__ __device__ double dist(const poly p_1, const poly p_2)
{
	return length(p_1.center(), p_2.center());
}

__host__ __device__ double dist(const poly p, const line l)
{
	double d = DBL_MAX;
	for (int i = 0; i < 20; i++)
	{
		if (is_cross(p[i], l))
		{
			return 0;
		}
		double t = l.point_dist(p[i].origin);
		if (t < d)
		{
			d = t;
		}
	}
	return d;
}

__host__ __device__ double dist(const poly p, const ray l)
{
	double d = DBL_MAX;
	for (int i = 0; i < 20; i++)
	{
		if (is_cross(p[i], l))
		{
			return 0;
		}
		double t = l.point_dist(p[i].origin);
		if (t < d)
		{
			d = t;
		}
	}
	return d;
}

__host__ __device__ double dist(const poly p, const seg l)
{
	double d = DBL_MAX;
	for (int i = 0; i < 20; i++)
	{
		if (is_cross(p[i], l))
		{
			return 0;
		}
		double t = l.point_dist(p[i].origin);
		if (t < d)
		{
			d = t;
		}
	}
	return d;
}
