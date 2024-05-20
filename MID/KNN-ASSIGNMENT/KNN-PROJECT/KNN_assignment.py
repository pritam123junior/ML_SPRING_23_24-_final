import numpy as np
import matplotlib.pyplot as plt
import data_utils
import download

def download_data():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    download_dir = "./data"
    download.maybe_download_and_extract(url,download_dir)

# Class to initialize and apply K-nearest neighbour classifier
class KNearestNeighbor(object):
    def __init__(self):
        pass

    # Method to initialize classifier with training data
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X, k=1, num_loops=0, distance='euclidean'):  # Added distance parameter
        if num_loops == 0:
            dists = self.compute_distances(X, distance)  # Pass distance parameter
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        return self.predict_labels(dists, k=k)

def compute_distances(self, X, distance='euclidean'):  # Added distance parameter
        num_test = X.shape[0]  # Number of test samples
        num_train = self.X_train.shape[0]  # Number of training samples
        dists = np.zeros((num_test, num_train))  # Initialize distance matrix with zeros
        for i in range(num_test):  # Iterate over each test sample
            if distance == 'euclidean':  # If distance metric is Euclidean
                dists[i, :] = np.sqrt(np.sum(np.square(self.X_train - X[i, :]), axis=1))  # Calculate Euclidean distance
            elif distance == 'manhattan':  # If distance metric is Manhattan
                dists[i, :] = np.sum(np.abs(self.X_train - X[i, :]), axis=1)  # Calculate Manhattan distance
            else:  # If distance metric is not supported
                raise ValueError('Invalid distance metric')  # Raise an error
        return dists  # Return computed distances

def predict_labels(self, dists, k=1):  # Predict labels using nearest neighbors
        num_test = dists.shape[0]  # Number of test samples
        y_pred = np.zeros(num_test)  # Initialize predicted labels
        for i in range(num_test):  # Iterate over each test sample
            closest_y = self.y_train[np.argsort(dists[i, :])[:k]]  # Find k nearest neighbors
            y_pred[i] = np.argmax(np.bincount(closest_y))  # Predict the label with most occurrences
        return y_pred  # Return predicted

def visualize_data(X_train, y_train):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()

if __name__ == "__main__":

    # Download CIFAR10 data and store it in the current directory if you have not done it.
    download_data()
    cifar10_dir = './data/cifar-10-batches-py'

    # Load training and testing data from CIFAR10 dataset
    X_train, y_train, X_test, y_test = data_utils.load_CIFAR10(cifar10_dir)

    # Checking the size of the training and testing data
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    # Memory error prevention by subsampling data. We sample 10000 training examples and 1000 test examples.
    num_training = 7000
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    num_test = 700
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Flatten the training and test data so each row consists of all pixels of an example
    X_train = np.reshape(X_train, (X_train.shape[0], -1))  # Flatten training data
    X_test = np.reshape(X_test, (X_test.shape[0], -1))  # Flatten test data
    print(X_train.shape, X_test.shape)  # X_train should be (10000, 3072) and X_test should be (1000, 3072)

    # Performing KNN
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)  # Train classifier
    # Use Euclidean distance
    y_test_pred_euclidean = classifier.predict(X_test, k=5, distance='euclidean')
    num_correct = np.sum(y_test_pred_euclidean == y_test)
    accuracy_euclidean = float(num_correct) / num_test
    print('Got %d / %d correct with k=5 =>accuracy_euclidean: %f'% (num_correct, num_test,accuracy_euclidean*100))

    # Use Manhattan distance
    y_test_pred_manhattan = classifier.predict(X_test, k=5, distance='manhattan')
    num_correct = np.sum(y_test_pred_manhattan == y_test)
    accuracy_manhattan = float(num_correct) / num_test
    print('Got %d / %d correct with k=5 =>accuracy_manhattan: %f'% (num_correct, num_test,accuracy_manhattan*100))

    # Perform 5-fold cross-validation to find the optimal k from choices below
    num_folds = 5
    k_choices = [1, 3, 5, 8, 10]

    X_train_folds = np.array_split(X_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)
    k_to_accuracies = {}  # dictionary to hold validation accuracies for each k

    for k in k_choices:
        k_to_accuracies[k] = []  # each key, k, should hold its list of 5 validation accuracies

        # For each fold of cross validation
        for i in range(num_folds):
        # Split training data into validation fold and training folds
            X_val_fold = X_train_folds[i]  # Validation data
            y_val_fold = y_train_folds[i]  # Validation labels
            X_train_fold = np.concatenate(X_train_folds[:i] + X_train_folds[i+1:])  # Training data
            y_train_fold = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:])  # Training labels

            # Initialize classifier with training folds and compute distances between examples in validation fold and training folds
            classifier.train(X_train_fold, y_train_fold)  # Train classifier
            dists_fold = classifier.compute_distances(X_val_fold)  # Compute distances

            # Use classifier to predict labels of validation fold for the given k value
            y_val_pred = classifier.predict_labels(dists_fold, k=k)  # Predict labels for validation data

            # Number of test examples correctly predicted, where y_val_pred contains labels predicted by classifier on validation fold
            num_correct = np.sum(y_val_pred == y_val_fold)
            accuracy = float(num_correct) / y_val_fold.shape[0]
            k_to_accuracies[k].append(accuracy)

    print("Printing our 5-fold accuracies for varying values of k:")
    print()
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))

    for k in sorted(k_to_accuracies):
        print('k = %d, avg. accuracy = %f' % (k, sum(k_to_accuracies[k])/5))

    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    # Plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.savefig('cross-validation_accuracy.jpg')

    # Choose the best value of k based on cross-validation results
    best_k = k_choices[np.argmax(accuracies_mean)]  # Choose k with highest mean accuracy

    # Intialize classifier and predict labels of test data, X_test, using the best value of k
    classifier = KNearestNeighbor()  # Initialize classifier
    classifier.train(X_train, y_train)  # Train classifier
    y_test_pred = classifier.predict(X_test, k=best_k)  # Predict labels for test data
    # Computing and displaying the accuracy for the best k found during cross-validation
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct on test data => accuracy: %f' % (num_correct, num_test, accuracy*100))
    # Accuracy above should be ~ 57-58%
