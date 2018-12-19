all: MnistTrainer MnistUI MnistTest
MnistTrainer: mnist_trainer.cpp */*.hpp */*.cpp
	g++ mnist_trainer.cpp */*.cpp -Ofast -o MnistTrainer -lm -lpthread
MnistUI: mnist_ui.cpp NN/NN.* NN/Util.hpp MNIST/Util.hpp
	g++ mnist_ui.cpp NN/NN.cpp -Ofast -o MnistUI -lm -lsfml-system -lsfml-window -lsfml-graphics
MnistTest: mnist_test.cpp NN/NN.* MNIST/Loader.* NN/Util.hpp MNIST/Util.hpp
	g++ mnist_test.cpp NN/NN.cpp MNIST/Loader.cpp -Ofast -o MnistTest -lm
