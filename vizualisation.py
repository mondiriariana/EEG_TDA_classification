import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.metrics import confusion_matrix
import os

def plot_confusion_matrix(y_true, y_pred, classes, title):
    """
    Plots a confusion matrix heatmap for model predictions versus true labels.

    Args:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
        classes (list): Class names for axis labels.
        title (str): Title of the plot.
    """

    confusion_mat = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Model predictions")
    plt.ylabel("True labels")
    plt.show()

def plot_class_distribution(df, label_column='y'):
    """
    Plot a pie chart showing the distribution of class labels.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the labels.
    - label_column (str): The column name for class labels. Default is 'y'.

    Displays:
    - A pie chart with percentage breakdown of each class.
    """
    value_counts = df[label_column].value_counts()

    custom_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']

    plt.figure(figsize=(10, 6))
    plt.pie(
        value_counts,
        labels=value_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        wedgeprops={'edgecolor': 'black'},
        pctdistance=0.8,
        colors=custom_colors,
    )
    plt.title('Class Distribution')
    plt.axis('equal')  
    plt.show()


def plot_persistence_images(labels, persistence_images, n_per_label=5):
    unique_labels = np.unique(labels)
    fig, axs = plt.subplots(len(unique_labels), n_per_label, figsize=(12, 12))
    
    for i, label in enumerate(unique_labels):
        idxs = np.where(labels == label)[0][:n_per_label]
        for j in range(n_per_label):
            ax = axs[i, j] if len(unique_labels) > 1 else axs[j]
            if j < len(idxs):
                img_vector = persistence_images[idxs[j]]
                pimg_h0 = img_vector[:100].reshape((10, 10))
                ax.imshow(pimg_h0, cmap='plasma', origin='lower')
                ax.axis('off')
                ax.set_title(f"Label {label}")
            else:
                ax.axis('off')
    
    plt.tight_layout()
    plt.show()