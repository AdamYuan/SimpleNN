#include "Trainer.hpp"
#include "Util.hpp"
#include <cfloat>
#include <random>
#include <algorithm>
#include <atomic>
#include <thread>
#include <future>

Trainer::Trainer(SimpleNN &nn, 
		float learning_rate, 
		unsigned iteration, unsigned batch_size)
	: 
		m_nn(nn),
		m_learning_rate(learning_rate), 
		m_iteration(iteration), 
		m_batch_size(batch_size)
{
	m_gradients.resize(m_nn.m_weigh_count);
}

void Trainer::init_adam()
{
	m_adam_m.resize(m_nn.m_weigh_count, 0.0);
	m_adam_v.resize(m_nn.m_weigh_count, 0.0);
	m_adam_beta1_t = kAdamBeta1;
	m_adam_beta2_t = kAdamBeta2;
}

void Trainer::init_thread()
{
	m_threads = std::thread::hardware_concurrency();
	m_thread_nn.resize(m_threads);
	for(auto &i : m_thread_nn) i = m_nn;
	m_thread_gradient_sum.resize(m_threads);
	for(auto &i : m_thread_gradient_sum) i.resize(m_nn.m_weigh_count);
}

void Trainer::init_batch(unsigned data_size)
{
	std::random_device rd;
	std::mt19937 rng{rd()};
	m_current_batch = 0;
	std::vector<unsigned> indices(data_size);
	for(unsigned i = 0; i < data_size; ++i) indices[i] = i;
	std::shuffle(indices.begin(), indices.end(), rng);

	unsigned batch_count = data_size / m_batch_size;
	if(data_size % m_batch_size) batch_count ++;

	m_batches.resize(batch_count);
	for(unsigned i = 0; i < m_batches.size(); ++i)
	{
		auto begin = indices.begin() + i*m_batch_size;
		auto end = indices.begin() + std::min((unsigned)data_size, (i+1)*m_batch_size);
		m_batches[i] = std::vector<unsigned>{begin, end};
	}
}

void Trainer::Run(const std::vector<TrainingData> &training_set)
{
	init_thread();
	init_adam();
	init_batch(training_set.size());

	printf("=====================TRAINING BEGIN==========================\n");
	printf("args: learning_rate = %lf batch_size = %d\n iteration = %d\n", m_learning_rate, m_batch_size, m_iteration);

	for(unsigned i = 0; i < m_iteration; ++i)
	{
		printf("\riteration #%d", i + 1);
		fflush(stdout);
		get_gradient(training_set);
		adam();
	}
	printf("\ntrained %d weights, mse = ", m_nn.m_weigh_count);
	fflush(stdout);
	printf("%lf\n", get_mse(training_set));
	printf("=====================TRAINING COMPLETE=======================\n");
}

void Trainer::get_gradient(const std::vector<TrainingData> &training_set) //2: return max gradient
{
	const auto &cur_batch = m_batches[m_current_batch];

	std::fill(m_gradients.begin(), m_gradients.end(), 0.0);

	{
		std::atomic_uint counter{0};
		unsigned cores = m_threads;
		std::vector<std::future<void>> futures;
		std::mutex mtx;

		while(cores --)
		{
			futures.push_back(
					std::async([&](unsigned thr_idx)
						{
							SimpleNN &nn = m_thread_nn[thr_idx];
							std::copy(m_nn.m_weights.data(), m_nn.m_weights.data() + m_nn.m_weigh_count, nn.m_weights.data());

							std::vector<float> &gs = m_thread_gradient_sum[thr_idx];
							std::fill(gs.begin(), gs.end(), 0.0);
							while(true)
							{
								unsigned i = counter ++;
								if(i >= cur_batch.size()) break;
								unsigned j = cur_batch[i];
								nn.Evaluate(training_set[j].m_inputs);
								nn.BackPropagation(training_set[j].m_expected, &gs);
							}

							mtx.lock();
							for(unsigned j = 0; j < m_nn.m_weigh_count; ++j)
								m_gradients[j] += gs[j];
							mtx.unlock();
						}, 
						cores)
					);
		}
	}

	float rt = 1.0 / (float)cur_batch.size();
	for(auto &i : m_gradients) i *= rt;

	m_current_batch = (m_current_batch + 1) % m_batches.size();
}

void Trainer::adam()
{
	for(unsigned i = 0; i < m_nn.m_weigh_count; ++i)
	{
		float &m = m_adam_m[i], &v = m_adam_v[i], &g = m_gradients[i];
		m = kAdamBeta1*m + (1.0 - kAdamBeta1)*g;
		v = kAdamBeta2*v + (1.0 - kAdamBeta2)*g*g;
		float hm = m / (1.0 - m_adam_beta1_t);
		float hv = v / (1.0 - m_adam_beta2_t);
		m_nn.m_weights[i] -= m_learning_rate * hm / (sqrtf(hv) + kAdamEps);
	}

	m_adam_beta1_t *= kAdamBeta1;
	m_adam_beta2_t *= kAdamBeta2;
}

float Trainer::get_mse(const std::vector<TrainingData> &training_set) const
{
	float mse{}, tmp;
	for(const auto &data : training_set)
	{
		m_nn.Evaluate(data.m_inputs);
		for(unsigned i = 0; i < m_nn.m_layers.back(); ++i)
		{
			tmp = data.m_expected[i] - m_nn.GetOutput()[i];
			mse += tmp*tmp;
		}
	}
	return mse / float(training_set.size());
}
