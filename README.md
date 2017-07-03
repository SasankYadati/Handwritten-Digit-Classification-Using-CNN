# Handwritten-Digit-Classification-Using-CNN
The objective of this project is to build a image-classifier using Convolutional Neural Networks to accurately categorize the handwritten digits. The data for this project can be found <a href="http://yann.lecun.com/exdb/mnist/">here</a> and the files are expected to be stored in the folder "/data/" relative to the repository.

# Convolutional Nerual Networks (CNNs)
Convolutional Neural Networks are a special type of neural networks which are very effective in image recognition and classification. They are powerful as they make use of spatial relationships among the features.
<br>
<br>
There are four main operations in a typical CNN.
<br>
<br>
1.Convolution
<br>
2.Non Linearity
<br>
3.Pooling / Sub Sampling / Down Sampling
<br>
4.Fully Connected Layer
<br>
<br>
To learn more about CNNs, go <a href="https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/">here</a>.

# Network Architecture
We use three types of layers, in this model. They are the convolutional layer, pooling layer and fully connected layer.
For this problem, I have defined the following network architecture.
<br>
<br>
Note: The input is a sample set of 60,000 images, where each image is 28x28 pixels with 1 channel.
<br>
<br>
The first layer is a Convolutional layer with 20 filters each of size (6,6) followed by another Convolutional layer with 20 filters each of size (3,3). Then, we have a Max Pooling layer of size (4,4).
<br>
<br>
Similarly, I have defined same set of layers with Convolutional layer having only 10 filters this time.
All the Convolutional layers defined above have ReLU as activation function.
<br>
<br>
Then there are fully connected layers with 30 units in the first and 10 (no. of o/p units) in the second.
The first layer uses tanh as activation function and the second one uses softmax.
<br>
<br>
I also introduced dropouts in between the stacks of layers to avoid overfitting. The key idea behind dropout is to randomly drop units along with their connections to prevent units from co-adapting too much. Read more <a href="https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf">here</a>.

# MNIST Data
The data can be found <a href="http://yann.lecun.com/exdb/mnist/">here</a>.

# Environment Setup
I recommend using Anaconda for running the script. Run the following command on conda shell to create a new environment with all the required packages.
<br>
<br>
<code>conda env create -f mnist-env.yml</code>
<br>
<br>
The mnist-env.yml file can be found in the repo.

# Training
Train the classifier by running the following command.
<br>
<br>
<code>python mnist-classifier.py</code>
<br>
<br>
I was able to achieve 97.14% accuracy on training data and 97.86% on test data. I encourage you to play with the architecture and parameters to see what you can achieve. Feel free to discuss the same.

# Classifying New Images

