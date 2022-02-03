from math import ceil, sqrt
from typing import Sequence

import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_dataset_sample(data_loader, classes_str, network=None):
    """ Show an image """
    images, labels = next(iter(data_loader))

    if network:
        outputs = network(images)
        _, predicted = torch.max(outputs, 1)
        labels_str = [f'{classes_str[label]}=>{classes_str[pred]}' for label, pred in zip(labels, predicted)]
        title = f'Sample of {len(images)} images\n[label]=>[Network inference]'
    else:
        labels_str = [classes_str[label] for label in labels]
        title = f'Sample of {len(images)} images'

    images = images / 2 + 0.5  # Un-normalize
    images = np.transpose(images.numpy(), (0, 2, 3, 1))  # Convert from Tensor image

    images_show(images, labels_str, title)


def images_show(images: Sequence, labels: Sequence[str] = None, title: str = '') -> None:
    """
    Plot a bunch of images in the same plot.

    :param images: List of array-like or PIL image to plot.
    :param labels: Label to plot for each image (ignored if missing).
    :param title: Global plot title (ignored if missing).
    """

    nb_img = len(images)

    if nb_img == 0:
        raise ValueError('No image to print')

    # Only on image, no need of subplot
    if nb_img == 1:
        plt.imshow(images[0], interpolation='none')
        plt.axis('off')
        title_str = ''
        if title:
            title_str = title
        if labels and len(labels) > 0 and labels[0]:
            if len(title_str) > 0:
                title_str += '\n' + labels[0]
            else:
                title_str = labels[0]
        if len(title_str) > 0:
            plt.title(title_str)

    # More than 1 image
    else:
        if nb_img < 4:
            # For 3 or below just plot them in one line
            nb_rows = 1
            nb_cols = nb_img
        else:
            nb_rows = nb_cols = ceil(sqrt(nb_img))
            nb_rows = ceil(nb_img / nb_rows)  # Remove empty rows if necessary

        # Create subplots
        fig, axs = plt.subplots(nrows=nb_rows, ncols=nb_cols, figsize=(nb_cols * 2, nb_rows * 2 + 1))

        for row in range(nb_rows):
            for col in range(nb_cols):
                i = row * nb_cols + col
                if nb_rows == 1:
                    ax = axs[col]
                else:
                    ax = axs[row, col]
                # Disable axis even if no image
                ax.axis('off')
                # Add image and image caption
                if i < len(images):
                    ax.imshow(images[i], interpolation='none')
                    if labels and len(labels) > i and labels[i]:
                        ax.set_title(labels[i])

        if title:
            fig.suptitle(title)

    # Adjust the padding between and around subplots
    plt.tight_layout()
    # Show it
    plt.show()
