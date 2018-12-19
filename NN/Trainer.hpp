#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "NN.hpp"

constexpr float kAdamBeta1{0.9}, kAdamBeta2{0.999}, kAdamEps{1e-8};

struct TrainingData { std::vector<float> m_inputs, m_expected; };

class Trainer
{
	private:
		SimpleNN &m_nn;
		float m_learning_rate;
		unsigned m_iteration, m_batch_size, m_current_batch;
		std::vector<std::vector<unsigned>> m_batches;
		void init_batch(unsigned data_size);

		std::vector<std::vector<float>> m_thread_gradient_sum;
		std::vector<float> m_gradients;

		std::vector<SimpleNN> m_thread_nn;
		unsigned m_threads;
		void init_thread();

		void get_gradient(const std::vector<TrainingData> &training_set);
		std::vector<float> m_adam_m, m_adam_v; float m_adam_beta1_t, m_adam_beta2_t;
		void init_adam();
		void adam();
		float get_mse(const std::vector<TrainingData> &training_set) const;

	public:
		Trainer(SimpleNN &nn, float learning_rate, unsigned iteration, unsigned batch_size);
		void Run(const std::vector<TrainingData> &training_set);
};

#endif
