# Code based on official pytorch tutorial:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
from collections import defaultdict

import torch
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets

from network import CNN
from utils.plotting import plot_dataset_sample

# Meta parameters
BATCH_SIZE = 4  # Number of image sent to the network input for each inference
NB_EPOCH = 2  # The number of time we will go through the whole dataset during the training
LEARNING_RATE = 0.001  # Learning rate for the stochastic gradient descent (SGD)
MOMENTUM = 0.9  # Momentum factor the stochastic gradient descent (SGD)
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def get_datasets():
    """ Load train and test datasets """
    # Transformation to apply to every data point
    # Transform to tensor, then normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load train dataset
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Load test dataset
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, test_loader


def train(network, train_loader):
    """ Train a neural network with a dataset """
    print(f'\nTraining started on {len(train_loader)} images during {NB_EPOCH} epochs ...')

    # Method to compute the loss
    criterion = CrossEntropyLoss()
    # Gradient descent configuration
    optimizer = SGD(network.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # Loop over the dataset multiple times
    for epoch in range(NB_EPOCH):
        running_loss = 0
        for i, data in enumerate(train_loader):
            # Get the inputs; data is a list of (inputs, labels)
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward (Inference)
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            # Backward
            loss.backward()
            # Optimize
            optimizer.step()

            # Print statistics every 2000 mini-batches
            running_loss += loss.item()
            if i % 2000 == 1999:
                progress = (epoch * len(train_loader) + i) / (NB_EPOCH * len(train_loader))
                print(f'{progress:6.2%} | Epoch {epoch + 1} - Batch {i + 1:5d} - Loss: {running_loss / 2000:.3f}')
                running_loss = 0

    print('Training completed\n')


def test(network, test_loader):
    """ Test a neural network with a dataset """
    correct = 0
    total = 0
    correct_pred = defaultdict(int)  # Dictionary with value 0 by default
    total_pred = defaultdict(int)

    # Disable some network features for testing (eg. dropout)
    network.eval()
    # Since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            # Get the inputs; data is a list of (inputs, labels)
            images, labels = data

            # Calculate outputs by running images through the network
            outputs = network(images)

            # The class with the highest energy is what we choose as prediction
            _, predictions = torch.max(outputs, 1)

            # Save result for accuracy
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            # Collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[CLASSES[label]] += 1
                total_pred[CLASSES[label]] += 1

    # Print accuracy result
    print(f'Accuracy of the network on the {len(test_loader)} test images: {correct / total:.2%}')
    for classname, correct_count in correct_pred.items():
        print(f'\t- {classname:5s} -> {correct_count / total_pred[classname]:.1%}')

    # Plot some image with network prediction
    plot_dataset_sample(test_loader, CLASSES, network)


def main():
    train_loader, test_loader = get_datasets()
    network = CNN()

    train(network, train_loader)
    test(network, test_loader)


if __name__ == '__main__':
    main()
