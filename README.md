# SimpleNN
A simple neural network implemented with pure c++, trained with ADAM optimizer, test with MNIST dataset (98.62% accuracy)

## Compile
```sh
make
./MnistTrainer mnist.nn #train neural network (multithreaded)
./MnistTest mnist.nn    #or bests/784-300-300-98_62.nn for the best result on my computer
./MnistUI bests/784-300-300-98_62.nn #a ui to recognize handwritten digit (written with SFML)
```

## MnistUI
### Usage
**DRAG** to write a digit  
**SCROLL** to change brush radius  
**PRESS SPACE** to recognize digit with neural network
### Screenshots
![](https://raw.githubusercontent.com/AdamYuan/SimpleNN/master/screenshots.gif)
