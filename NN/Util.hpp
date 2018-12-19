#ifndef UTIL_HPP
#define UTIL_HPP

#include <cmath>

/*inline float sigmoid(float x)
{
	return 1.0 / (1.0 + expf(-x));
}
inline float sigmoid_derivative(float x)
{
	float s = sigmoid(x);
	return s*(1.0 - s);
}*/

inline float relu(float x)
{
	return x < 0.0 ? 0.0 : x;
}

inline float relu_derivative(float x)
{
	return x < 0.0 ? 0.0 : 1.0;
}
/*
inline float activation(float x)
{
	return relu(x);
}
inline float activation_derivative(float x)
{
	return relu_derivative(x);
}*/

#endif
