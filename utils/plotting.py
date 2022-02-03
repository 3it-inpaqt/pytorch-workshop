from math import ceil, sqrt
from typing import Iterable, List, Sequence

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

# Set plot style.
sns.set_theme(rc={
    'axes.titlesize': 15,
    'figure.titlesize': 18,
    'axes.labelsize': 13,
    'figure.autolayout': True
})


def plot_dataset_sample(data_loader, classes_str, network=None):
    """ Show an image """
    images, labels = next(iter(data_loader))

    if network:
        outputs = network(images)
        _, predicted = torch.max(outputs, 1)
        labels_str = [f'{classes_str[label]}=>{classes_str[pred]}' for label, pred in zip(labels, predicted)]
        title = f'Sample of {len(images)} images\n[label]=>[inference]'
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


def plot_confusion_matrix(nb_labels_predictions: np.ndarray, class_names: Iterable[str] = None,
                          annotations: bool = True) -> None:
    """
    Plot the confusion matrix for a set a predictions.

    :param nb_labels_predictions: The count of prediction for each label.
    :param class_names: The list of readable classes names
    :param annotations: If true the accuracy will be written in every cell
    """

    overall_accuracy = nb_labels_predictions.trace() / nb_labels_predictions.sum()
    rate_labels_predictions = nb_labels_predictions / nb_labels_predictions.sum(axis=1).reshape((-1, 1))

    sns.heatmap(rate_labels_predictions,
                vmin=0,
                vmax=1,
                square=True,
                fmt='.0%',
                cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto',
                annot=annotations,
                cbar=(not annotations))
    plt.title(f'Confusion matrix of {len(nb_labels_predictions)} classes '
              f'with {overall_accuracy * 100:.2f}% overall accuracy')
    plt.xlabel('Predictions')
    plt.ylabel('Labels')

    # Adjust the padding between and around subplots
    plt.tight_layout()
    # Show it
    plt.show()


def plot_train_progress(loss_evolution: List[float], batch_size: int, accuracy_evolution: List[dict] = None,
                        batch_per_epoch: int = 0, best_checkpoint: dict = None) -> None:
    """
    Plot the evolution of the loss and the accuracy during the training.

    :param loss_evolution: A list of loss for each batch.
    :param batch_size: The size of the batch.
    :param accuracy_evolution: A list of dictionaries as {batch_num, validation_accuracy, train_accuracy}.
    :param batch_per_epoch: The number of batch per epoch to plot x ticks.
    :param best_checkpoint: A dictionary containing information about the best version of the network according to
        validation score processed during checkpoints.
    """
    with sns.axes_style("ticks"):
        fig, ax1 = plt.subplots()

        # Vertical lines for each batch
        if batch_per_epoch:
            if len(loss_evolution) / batch_per_epoch > 400:
                batch_per_epoch *= 100
                label = '100 epochs'
            elif len(loss_evolution) / batch_per_epoch > 40:
                batch_per_epoch *= 10
                label = '10 epochs'
            else:
                label = 'epoch'

            for epoch in range(0, len(loss_evolution) + 1, batch_per_epoch):
                # Only one with label for clean legend
                ax1.axvline(x=epoch, color='black', linestyle=':', alpha=0.2, label=label if epoch == 0 else '')

        # Plot loss
        ax1.plot(loss_evolution, label='loss', color='tab:gray')
        ax1.set_ylabel('Loss')
        ax1.set_ylim(bottom=0)

        if accuracy_evolution:
            legend_y_anchor = -0.25

            # Plot the accuracy evolution if available
            ax2 = plt.twinx()
            checkpoint_batches = [a['batch_num'] for a in accuracy_evolution]
            ax2.plot(checkpoint_batches, [a['train_accuracy'] for a in accuracy_evolution],
                     label='train accuracy',
                     color='tab:orange')
            ax2.plot(checkpoint_batches, [a['validation_accuracy'] for a in accuracy_evolution],
                     label='validation accuracy',
                     color='tab:green')

            # Star marker for best validation accuracy
            if best_checkpoint and best_checkpoint['batch_num'] is not None:
                ax2.plot(best_checkpoint['batch_num'], best_checkpoint['validation_accuracy'], color='tab:green',
                         marker='*', markeredgecolor='k', markersize=10, label='best valid. accuracy')
                legend_y_anchor -= 0.1

            ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            ax2.set_ylabel('Accuracy')
            ax2.set_ylim(bottom=0, top=1)

            # Place legends at the bottom
            ax1.legend(loc="lower left", bbox_to_anchor=(-0.1, legend_y_anchor))
            ax2.legend(loc="lower right", bbox_to_anchor=(1.2, legend_y_anchor))
        else:
            # Default legend position if there is only loss
            ax1.legend()

        plt.title('Training evolution')
        ax1.set_xlabel(f'Batch number (size: {batch_size:n})')

        # Adjust the padding between and around subplots
        plt.tight_layout()
        # Show it
        plt.show()
