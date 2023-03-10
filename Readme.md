# neural network with rust

Here, we define a NeuralNetwork struct that stores the weights and biases of the network as vectors of ndarrays. The new() method creates a new instance of NeuralNetwork with random weights and biases based on the layer sizes passed in as an argument.

We also define a sigmoid() function that applies the sigmoid activation function to a given scalar value, and a feedforward() method that takes an input vector, applies the weights and biases of the network, and returns an output vector.

In the main() function, we create a new instance of NeuralNetwork with layer sizes [2, 3, 1], and feed it an input vector [1.0, 0.0]. We then print the input and output vectors for inspection. Note that this is just a simple example, and a real-world neural network would likely have more complex architectures, activation functions, and learning algorithms.