import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Generate a synthetic dataset (you can replace this with your actual data)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

class NeuralNetwork(object):
    def __init__(self):
        inputLayerNeurons = 2
        hiddenLayerNeurons = 10
        outLayerNeurons = 5  # Update to 5 neurons for multi-class classification

        self.learning_rate = 0.5
        self.W_HI = np.random.randn(inputLayerNeurons, hiddenLayerNeurons)
        self.W_OH = np.random.randn(hiddenLayerNeurons, outLayerNeurons)

    def sigmoid(self, x, der=False):
        if der:
            return x * (1 - x)  # Derivative of sigmoid function
        else:
            return 1 / (1 + np.exp(-x))  # Sigmoid activation function

    def feedForward(self, X):
        hidden_input = np.dot(X, self.W_HI)
        self.hidden_output = self.sigmoid(hidden_input)

        output_input = np.dot(self.hidden_output, self.W_OH)
        pred = self.sigmoid(output_input)
        return pred

    def backPropagation(self, X, Y, pred):
        output_error = Y - pred
        output_delta = self.learning_rate * output_error * self.sigmoid(pred, der=True)

        hidden_error = output_delta.dot(self.W_OH.T)
        hidden_delta = self.learning_rate * hidden_error * self.sigmoid(self.hidden_output, der=True)

        self.W_HI += X.T.dot(hidden_delta)  # Update weights for hidden layer
        self.W_OH += self.hidden_output.T.dot(output_delta)  # Update weights for output layer

    def train(self, X, Y):
        output = self.feedForward(X)
        self.backPropagation(X, Y, output)
    def evaluate(self, X_test, Y_test):
    # Get predictions for the test data
        predictions = self.feedForward(X_test)

    # Convert predictions to binary (0 or 1) based on a threshold (e.g., 0.5)
        binary_predictions = (predictions >= 0.5).astype(int)

    # Calculate metrics
        true_positives = np.sum(Y_test * binary_predictions)  # Correctly predicted positive cases
        false_positives = np.sum((1 - Y_test) * binary_predictions)  # Incorrectly predicted positive cases
        false_negatives = np.sum(Y_test * (1 - binary_predictions))  # Missed positive cases

    # Compute evaluation metrics
        accuracy = (true_positives + np.sum(1 - Y_test)) / len(Y_test)  # Overall accuracy
        precision = true_positives / (true_positives + false_positives)  # Precision
        recall = true_positives / (true_positives + false_negatives)  # Recall
        f1_score = 2 * (precision * recall) / (precision + recall)  # F1-score
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1_score)
# Create an instance of the neural network
NN = NeuralNetwork()

X_test = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
Y_test = np.array([[0], [1], [1], [0]])
NN.evaluate(X_test, Y_test)

err = []
for i in range(10000):
    NN.train(X, Y)
    err.append(np.mean(np.square(Y - NN.feedForward(X))))  # Mean squared error

# Plot the training error
plt.plot(err)
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.title("Training Error")
plt.show()

# Test the network on some examples
print("Predictions for [0, 0]:", NN.feedForward([0, 0]))
print("Predictions for [1, 1]:", NN.feedForward([1, 1]))
print("Predictions for [1, 0]:", NN.feedForward([1, 0]))
print("Predictions for [0, 1]:", NN.feedForward([0, 1]))

