# Machine Learning Architectures

I've been learning about types of machine learning/neural network architectures for my internship, so this is a small collection of the ones I've learned/currently learning.

## scratch_mlp_skl.py
Simple 3 layer MLP written from scratch with some help from the SciKit-Learn module. Classifier for the MNIST dataset (8x8 images). 

Architecture: Input(64) -> HiddenLayer(30) -> Output(10)

As a result of writing this neural net, I learned the general structure and math behind neural nets, including feed forward, back propagation, gradient descent, and activation and cost functions. 

## mlp_tf.py
Simple 3 layer MLP written using the TensorFlow module. Classifier for the MNIST dataset (28x28 images).

Architecture: Input(784) -> HiddenLayer(300) -> Output(10)

As a result of writing this neural net, I learned how to use TensorFlow to implement a simple MLP neural net. I also learned about tf's GradientDescent optimizer.

## cnn_tf.py
Convolutional neural network written using the TensorFlow module. Classifier for the MNIST dataset (28x28 images).

Architecture: Input(28x28x1) -> Conv+ReLU(28x28x32) -> MaxPool(14x14x32) -> Conv+ReLU(14x14x64) -> MaxPool(7x7x64) -> FC(3136x1000x10)

I decided to learn about CNN's because I wanted to create a license plate detector during my internship. I learned about the convolution and pooling operations, as well as different activation functions such as ReLU and Softmax. I also learned about the Adam optimizer.

## cnn_keras.py
Convolutional neural network written using the Keras module. Classifier for the MNIST dataset (28x28 images).

Architecture: Input(28x28x1) -> Conv+ReLU(28x28x32) -> MaxPool(14x14x32) -> Conv+ReLU(14x14x64) -> MaxPool(7x7x64) -> FC(3136x1000x10)

I implemented the same architecture as in cnn_tf.py using Keras, which drastically reduced the amount of code I needed to write. I like the level of abstraction that Keras provides the user. 

I included a graph of the model's accuracy in /model_graphs/cnn_keras_model_accuracy.png
