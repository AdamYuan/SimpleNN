#include "NN.hpp"
#include "Util.hpp"
#include <iomanip>
#include <fstream>
#include <numeric>
#include <cstdio>

SimpleNN::SimpleNN(const std::vector<unsigned> &layers)
{
	Initialize(layers);
}

SimpleNN::SimpleNN(const char *filename)
{
	Load(filename);
}

void SimpleNN::Initialize(const std::vector<unsigned> &layers)
{
	if(layers.size() < 2)
	{
		printf("layer count less than 2\n");
		return;
	}
	m_layers = layers;
	m_neuron_count = std::accumulate(m_layers.begin(), m_layers.end(), 0u);

	m_neurons.m_values.resize(m_neuron_count);
	m_neurons.m_linear_values.resize(m_neuron_count);
	m_neurons.m_gradients.resize(m_neuron_count);
	m_neurons.m_forward_values.resize(m_neuron_count);
	m_neurons.m_forward_weights.resize(m_neuron_count);
	m_layer_count = layers.size();

	std::vector<std::pair<float *, float *>> ranges(m_layers.size());
	for(unsigned s = 0, i = 0; i < m_layer_count; ++i)
	{
		ranges[i] = {m_neurons.m_values.data() + s, m_neurons.m_values.data() + s + m_layers[i]};
		s += m_layers[i];
	}
	for(unsigned s = m_layers.front(), i = 1; i < m_layer_count; ++i)
		for(unsigned j = 0; j < m_layers[i]; ++j, ++s)
			m_neurons.m_forward_values[s] = ranges[i-1];

	//Initialize weigh array
	m_weigh_count = 0;
	for(unsigned i = m_layers[0]; i < m_neuron_count; ++i)
		m_weigh_count += m_neurons.m_forward_values[i].second - m_neurons.m_forward_values[i].first + 1; //add one for bias
	m_weights.resize(m_weigh_count);
	//point weight to neurons
	m_weigh_count = 0;
	for(unsigned i = m_layers[0]; i < m_neuron_count; ++i)
	{
		m_neurons.m_forward_weights[i] = m_weights.data() + m_weigh_count;
		m_weigh_count += m_neurons.m_forward_values[i].second - m_neurons.m_forward_values[i].first + 1; //add one for bias
	}
}

void SimpleNN::Load(const char *filename)
{
	std::ifstream in{filename};
	in >> m_layer_count;
	std::vector<unsigned> layers;
	while(m_layer_count --)
	{
		unsigned s; in >> s;
		layers.push_back(s);
	}
	Initialize(layers);
	for(float &i : m_weights)
		in >> i;
	printf("loaded SimpleNN from %s\n", filename);
}

void SimpleNN::Save(const char *filename) const
{
	std::ofstream out{filename};
	out << m_layer_count << std::endl;
	for(unsigned i : m_layers) out << i << std::endl;
	for(float i : m_weights) out << std::setprecision(9) << i << std::endl;
	printf("saved SimpleNN to %s\n", filename);
}

void SimpleNN::HeRandomizeWeights()
{
	std::random_device rd; std::mt19937 rng{rd()};
	std::normal_distribution<float> rn{};
	for(unsigned i = m_layers.front(); i < m_neuron_count; ++i)
	{
		unsigned sz = m_neurons.m_forward_values[i].second 
			- m_neurons.m_forward_values[i].first + 1;
		float *p = m_neurons.m_forward_weights[i], *end = p + sz;
		//printf("%u\n", sz);
		for(; p != end; ++p)
			*p = rn(rng) * sqrtf(2.0 / sz);
	}
}

void SimpleNN::UniformRandomizeWeights(float wmin, float wmax)
{
	assert(wmin < wmax);
	std::random_device rd; std::mt19937 rng{rd()};
	std::uniform_real_distribution<float> dis{wmin, wmax};
	for(auto &i : m_weights)
		i = dis(rng);
}
