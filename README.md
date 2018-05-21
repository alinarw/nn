# Neural Networks - 1, 2, 3 Hidden Layers
Exploring the neural network training from the perspective of maximum likelihood and maximum a posteriori parameter learning.
1. Demonstrating that a neural network to maximize the log likelihood of observing the training data is one that has softmax output nodes and minimizes the criterion function of the negative log probability of training data set.
2. Parameters for Neural Network: 
- 1 hidden layer of 30 sigmoid nodes, and an output 10 softmax nodes from 1000 training images (100 images per digit) 
- Training the network for 30 complete epochs, using mini-batches of 10 training examples at a time 
- Learning rate = 0.1 
- Plotting the training error, testing error, criterion function on training data set, criterion function on testing data set of a separate 1000 testing images (100 images per digit), and the learning speed of the hidden layer 
- 2 hidden layers of 30 sigmoid nodes each 
- 3 hidden layers of 30 sigmoid nodes each 
- Performing the above with and without L2 regularization, λ = 5 
#### Tools & Liblaries: Python 3, Tensorﬂow, Numpy, Matplotlib
