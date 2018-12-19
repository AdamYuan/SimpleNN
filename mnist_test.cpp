#include <iostream>
#include <algorithm>
#include "NN/NN.hpp"
#include "MNIST/Loader.hpp"

int main(int argc, char **argv)
{
	if(argc != 2)
	{
		printf("Usage: ./MnistTest [snn filename]\n");
		return EXIT_FAILURE;
	}
	SimpleNN snn(argv[1]);

	unsigned correct_count = 0;
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
