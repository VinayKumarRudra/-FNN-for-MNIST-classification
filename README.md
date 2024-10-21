## MNIST Classification with FNN, Dropout, and BatchNorm

This project implements a *Feedforward Neural Network (FNN)* using *PyTorch* to classify digits from the *MNIST dataset. The model includes **Dropout* and *Batch Normalization* to improve generalization and stability during training. It trains for a number of epochs and plots the training and test losses over time.

### Model Architecture
The Feedforward Neural Network (FNN) consists of the following layers:
- *Input Layer*: The input consists of 28x28 pixel images, flattened to a vector of size 784.
- *First Hidden Layer*: A fully connected (dense) layer with 128 neurons, followed by:
  - *Batch Normalization*: This layer normalizes the input to improve training stability.
  - *ReLU Activation*: Non-linearity to enable the network to learn complex patterns.
  - *Dropout Layer*: Dropout with a probability of 0.5 to prevent overfitting.
- *Second Hidden Layer*: A fully connected layer with 64 neurons and a ReLU activation function.
- *Output Layer*: A fully connected layer with 10 output neurons, corresponding to the 10 digit classes (0–9), followed by a softmax function to output probabilities.

### Dataset
- The dataset used is *MNIST*, which consists of 70,000 grayscale images of handwritten digits, 28x28 pixels each. There are 60,000 training images and 10,000 test images.
- The dataset is loaded using torchvision.datasets and transformed with normalization.

### Dependencies
To run the project, you need the following Python libraries:
- *torch*: PyTorch library for deep learning.
- *torchvision*: For loading and processing the MNIST dataset.
- *matplotlib*: For plotting the training and test loss curves.
- *numpy*: For general numerical operations.

You can install these dependencies using:
bash
pip install torch torchvision matplotlib numpy


### How to Run the Code

1. *Clone the Repository*:
   Clone this repository to your local machine using:
   bash
   git clone <repository_link>
   cd <repository_directory>
   

2. *Install Dependencies*:
   Install the required packages using the following command:
   bash
   pip install -r requirements.txt
   

3. *Run the Code*:
   Run the script mnist_fnn.py to train the model and plot the training/test loss curves:
   bash
   python mnist_fnn.py
   

4. *Training Process*:
   - The model will be trained on the MNIST dataset for 10 epochs (you can change the number of epochs in the code).
   - During each epoch, the model computes the training loss and test loss.
   - After training, a plot will be generated showing the training and test loss over the number of epochs.

### Output

- After the training process is complete, the script will display a plot that contains:
  - *Training Loss vs. Epochs*: A blue line representing the training loss at each epoch.
  - *Test Loss vs. Epochs*: A red line representing the test loss at each epoch.
  
The plot helps visualize how the model's performance improves over time and how well it generalizes to unseen data.

### Code Structure

bash
.
├── mnist_fnn.py        # Main Python script that implements the model and training
├── README.md           # This README file
└── requirements.txt    # Required dependencies (optional)


### Additional Information

- The network includes both *Dropout* and *Batch Normalization* to improve generalization and avoid overfitting.
- The *Adam optimizer* is used for training with a learning rate of 0.001, and the *Cross Entropy Loss* function is used to calculate the loss.
- The dataset is split into 60,000 training images and 10,000 test images, and both training and test losses are plotted after every epoch.

### Example Results
Below is an example of the training and test loss plot that you should see after running the model:

![Training and Test Loss Plot](training_test_loss_plot.png)

### References
This implementation is built using the PyTorch framework and follows standard techniques for training feedforward neural networks on image data. The code has been adapted and customized for educational purposes based on the official PyTorch tutorials.
