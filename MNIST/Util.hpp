#ifndef MNIST_UTIL_HPP
#define MNIST_UTIL_HPP
#include <vector>
//transform the image from
//example:
//0 1 0 0    1 0 0 0
//0 1 0 0 to 1 0 0 0
//0 1 1 0    1 1 1 1
//0 0 1 0    0 0 0 1
//something like that ...
inline float lerp(float a, float b, float f) { return a+(b-a)*f; } 
inline void width_normalize(std::vector<float> *img)
{
	unsigned min_x = 28, min_y = 28, max_x = 0, max_y = 0;
	for(unsigned i = 0; i < 784; ++i)
		if((*img)[i] >= 0.1)
		{
			unsigned x = i % 28, y = i / 28;
			min_x = std::min(min_x, x);
			min_y = std::min(min_y, y);
			max_x = std::max(max_x, x);
			max_y = std::max(max_y, y);
		}
	if(min_x > max_x || min_y > max_y) return;
	std::vector<float> out(784);
	unsigned sub_x = max_x - min_x + 1, sub_y = max_y - min_y + 1;
	float x_mul = float(sub_x) / 28.0;
	float y_mul = float(sub_y) / 28.0;
	for(unsigned i = 0; i < 784; ++i)
	{
		unsigned x = i % 28, y = i / 28;
		float tx = x * x_mul, ty = y * y_mul;
		unsigned ux = tx, uy = ty;
		ux = std::min(ux, sub_x - 1);
		uy = std::min(uy, sub_y - 1);
		unsigned rx = min_x + ux, ry = min_y + uy;
		unsigned rx1 = std::min(rx + 1, 27u);
		unsigned ry1 = std::min(ry + 1, 27u);
		float s0 = (*img)[ry*28 + rx];
		float s1 = (*img)[ry*28 + rx1];
		float s2 = (*img)[ry1*28 + rx];
		float s3 = (*img)[ry1*28 + rx1];

		out[i] = lerp( lerp(s0, s1, tx - ux), lerp(s2, s3, tx - ux), ty - uy );
	}
	*img = out;
}
#endif
