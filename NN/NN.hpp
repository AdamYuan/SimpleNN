#ifndef NN_HPP
#define NN_HPP

#include <vector>
#include <cassert>
#include <random>
#include "Util.hpp"

class SimpleNN
{
	friend class Trainer;
	private:
	struct 
	{
		std::vector<float*> m_forward_weights;
		//forward_weights -> the weight for forward layer
		//including the bias
		std::vector<std::pair<float *, float *>> m_forward_values;
		//forward_value -> the neuron values in forward layer
		//(begin and end pointer)
		std::vector<float> m_linear_values, m_values, m_gradients;
		//value -> neuron value
		//for back propagation:
		//linear_value -> neuron value before applying activation function
		//gradients -> gradient for neurons in cost function
	} m_neurons; //define neural network
	std::vector<float> m_weights;
	std::vector<unsigned> m_layers;
	unsigned m_layer_count, m_neuron_count, m_weigh_count;

	public:
	SimpleNN() = default;
	SimpleNN(const std::vector<unsigned> &layers);
	SimpleNN(const char *filename);
	SimpleNN &operator = (const SimpleNN &r)
	{
		m_layer_count = r.m_layer_count;
		m_neuron_count = r.m_neuron_count;
		m_weigh_count = r.m_weigh_count;
		m_neurons = r.m_neurons;
		m_weights = r.m_weights;
		m_layers = r.m_layers;
		for(auto &i : m_neurons.m_forward_weights)
			i = i - r.m_weights.data() + m_weights.data();

		for(auto &i : m_neurons.m_forward_values)
		{
			i.first = i.first - r.m_neurons.m_values.data() + m_neurons.m_values.data();
			i.second = i.second - r.m_neurons.m_values.data() + m_neurons.m_values.data();
		}
		return *this;
	}
	void Initialize(const std::vector<unsigned> &layers);
	void UniformRandomizeWeights(float wmin, float wmax);
	void HeRandomizeWeights();
	inline void Evaluate(const std::vector<float> &input)
	{
		assert(input.size() == m_layers.front());
		std::copy(input.data(), input.data() + m_layers.front(), m_neurons.m_values.data());
		//for(unsigned i = 0; i < m_layers.front(); ++i)
		//	m_neurons.m_values[i] = input[i];
		for(unsigned i = m_layers.front(); i < m_neuron_count; ++i)
		{
			float &linear_val = m_neurons.m_linear_values[i], //linear value reference
				   *w = m_neurons.m_forward_weights[i];        //the weigh pointer
			linear_val = 0;
			for(float *j = m_neurons.m_forward_values[i].first,  //loop over all forward connected neurons
					*end = m_neurons.m_forward_values[i].second;
					j != end; ++j, ++w)
				linear_val += *j * *w;
			linear_val += *w; //bias
			m_neurons.m_values[i] = relu(linear_val);
		}
	}
	inline void BackPropagation(const std::vector<float> &expected, std::vector<float> *weigh_gradient_sum)
	{
		assert(expected.size() == m_layers.back());
		assert(weigh_gradient_sum->size() == m_weigh_count);

		//reset neuron gradients except for the last layer
		std::fill(m_neurons.m_gradients.begin(),
				m_neurons.m_gradients.begin() + m_neuron_count - m_layers.back(), 0);

		//set the gradients for last layer
		for(unsigned i = m_neuron_count - m_layers.back(), j = 0; i < m_neuron_count; ++i, ++j)
			m_neurons.m_gradients[i] = 2.0 * (m_neurons.m_values[i] - expected[j]);

		for(unsigned i = m_neuron_count - 1; i >= m_layers.front(); --i)
		{
			float delta_cost_over_z = m_neurons.m_gradients[i] * 
				relu_derivative(m_neurons.m_linear_values[i]);

			float *w = m_neurons.m_forward_weights[i];        //the weigh pointer
			float *j = m_neurons.m_forward_values[i].first;   //the value pointer
			int w_idx = w - m_weights.data();   //the weigh index
			int j_idx = j - m_neurons.m_values.data(); //the value idx

			for(float *end = m_neurons.m_forward_values[i].second; j != end; 
					++j, ++j_idx, ++w, ++w_idx)
			{
				(*weigh_gradient_sum)[w_idx] += *j * delta_cost_over_z;
				m_neurons.m_gradients[j_idx] += *w * delta_cost_over_z;
			}
			(*weigh_gradient_sum)[w_idx] += delta_cost_over_z; //process bias
		}
	}
	inline const float *GetOutput() const { return m_neurons.m_values.data() + m_neuron_count - m_layers.back(); }

	void Save(const char *filename) const;
	void Load(const char *filename);
};

#endif
