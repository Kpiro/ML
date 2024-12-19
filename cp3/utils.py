import numpy as np
import matplotlib.pyplot as plt
import math
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

random.seed(42)

def plot_samples_with_labels(data, true_labels, predicted_labels =[], num_samples=5, title = None, cmap='gray', randomize=True):
    """
    Function to plot a sample of images from a given collection along with their true and predicted labels.
    
    Parameters:
    - data: List or numpy array of data (images or any other type of data)
    - true_labels: List or numpy array of true labels corresponding to the data
    - predicted_labels: List or numpy array of predicted labels corresponding to the data
    - num_samples: Number of samples to plot (default is 5)
    - title: Title for the plot (optional)
    - cmap: Colormap for displaying images (default is 'gray')
    - randomize: Boolean flag to decide whether to sample randomly (True) or use the first items (False)
    """
    # Ensure that the number of samples does not exceed the number of data points
    num_samples = min(num_samples, len(data))
    random.seed(42)

    if randomize:
        # Randomly sample indices if randomize is True
        sample_indices = np.random.choice(len(data), num_samples, replace=False)
    else:
        # Use the first 'num_samples' items if randomize is False
        sample_indices = np.arange(num_samples)

    # Select the corresponding data and labels based on the sampled indices
    selected_data = np.array(data)[sample_indices]
    sample_true_labels = np.array(true_labels)[sample_indices]

    sample_predicted_labels = np.array(predicted_labels)[sample_indices] if len(predicted_labels) else []

    # Set the maximum number of columns to 5
    num_cols = 5
    num_rows = int(math.ceil(num_samples / num_cols))  # Number of rows based on the number of columns
    
    # Plot the selected samples in a grid layout
    plt.figure(figsize=(15, 3 * num_rows))  # Adjust the figure size based on the number of rows
    for i in range(num_samples):
        plt.subplot(num_rows, num_cols, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(selected_data[i], cmap=cmap)
        plt.axis('off')  # Remove axes
        plt.title(f"True: {sample_true_labels[i]}\nPred: {sample_predicted_labels[i] if len(sample_predicted_labels) else '-' }")
    
    if title:
        plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    # # Plot the selected samples
    # fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    # if num_samples == 1:
    #     axes = [axes]  # Make sure axes is always iterable
    # for i in range(num_samples):
    #     axes[i].imshow(selected_data[i], cmap=cmap)
    #     axes[i].axis('off')
    #     axes[i].set_title(f"True: {sample_true_labels[i]}\nPred: {sample_predicted_labels[i] if len(sample_predicted_labels) else '-' }")

    # if title:
    #     plt.suptitle(title, fontsize=16)
    # plt.tight_layout()
    # plt.show()


def classify(test_images, model):
    """
    Predicts the class labels for a batch of test images using a trained model.

    Args:
        test_images (numpy.ndarray): Preprocessed test images ready for prediction.
        model (tensorflow.keras.Model): A trained model to classify the images.

    Returns:
        numpy.ndarray: An array of predicted class labels.
    """
    predictions = model.predict(test_images)    # Predict probabilities
    predicted_class = np.argmax(predictions, axis=-1)    # Get class with highest probability
    return predicted_class



def plot_conf_matrix(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()