#include <cstdio>
#include <algorithm>
#include <cfloat>
#include "NN/Trainer.hpp"
#include "MNIST/Loader.hpp"

int main(int argc, char **argv)
{
	if(argc != 2)
	{
		printf("Usage: ./MnistTrainer [output snn filename]\n");
		return EXIT_FAILURE;
	}
	MnistLoader training_set{"MNIST/train-images.idx3-ubyte", "MNIST/train-labels.idx1-ubyte"};
	SimpleNN snn({784, 200, 200, 10});
	snn.UniformRandomizeWeights(0.0, 0.00001);
	Trainer{snn, 
		0.001,      //learning rate
		12000,    //20 epoches
		100,     //batch size
	}.Run(training_set.GetDataSet());
	snn.Save(argv[1]);
	//SimpleNN snn{"./bests/784-200-200-10_0.nn"};

	unsigned correct_count = 0;
	for(const auto &i : training_set.GetDataSet())
	{
		snn.Evaluate(i.m_inputs);
		int res = std::max_element(snn.GetOutput(), snn.GetOutput() + 10) - snn.GetOutput();
		if(i.m_expected[res] > 0.5) ++correct_count;
	}
	printf("training set accuracy: %lf%%\n", 100.0 * (float)correct_count / (float)training_set.GetDataSet().size());

	correct_count = 0;
	MnistLoader validation_set{"MNIST/t10k-images-idx3-ubyte", "MNIST/t10k-labels-idx1-ubyte"};
	for(const auto &i : validation_set.GetDataSet())
	{
		snn.Evaluate(i.m_inputs);
		int res = std::max_element(snn.GetOutput(), snn.GetOutput() + 10) - snn.GetOutput();
		if(i.m_expected[res] > 0.9) ++correct_count;
	}
	printf("validation set accuracy: %lf%%\n", 100.0 * (float)correct_count / (float)validation_set.GetDataSet().size());
	return EXIT_SUCCESS;
}
