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
inline float sample(const std::vector<float> &img, float x, float y) //x, y in [0, 28)
{
	unsigned ix = x, iy = y, ix1 = ix + 1, iy1 = iy + 1;
	float fx = x - ix, fy = y - iy;
	bool vx = 0 <= ix && ix < 28u;
	bool vy = 0 <= iy && iy < 28u;
	bool vx1 = 0 <= ix1 && ix1 < 28u;
	bool vy1 = 0 <= iy1 && iy1 < 28u;
	float s0 = vx && vy ? img[iy*28 + ix] : 0.0;
	float s1 = vx1 && vy ? img[iy*28 + ix1] : 0.0;
	float s2 = vx && vy1 ? img[iy1*28 + ix] : 0.0;
	float s3 = vx1 && vy1 ? img[iy1*28 + ix1] : 0.0;

	return lerp( lerp(s0, s1, fx), lerp(s2, s3, fx), fy );
}
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
		float tx = x * x_mul + min_x, ty = y * y_mul + min_y;
		out[i] = sample(*img, tx, ty);
	}
	*img = out;
}
#endif
