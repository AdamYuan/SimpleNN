# SimpleNN
A simple neuron network implemented with pure c++, trained with ADAM optimizer, test with MNIST dataset (98.13% accuracy)
```sh
make
./MnistTrainer mnist.nn #train neural network (multithreaded)
./MnistTest mnist.nn    #or bests/784-200-200-10_0.nn for the best result in my computer
./MnistUI mnist.nn      #a ui to recognize handwritten digit (written with SFML)
```
