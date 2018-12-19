#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include "../NN/Trainer.hpp"

class MnistLoader
{
	private:
		std::vector<TrainingData> m_data_set;
	public:
		MnistLoader() = default;
		MnistLoader(const char *file1, const char *file2);
		void Load(const char *file1, const char *file2);
		const std::vector<TrainingData> &GetDataSet() { return m_data_set; }
};

#endif
