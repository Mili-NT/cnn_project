# Import required libraries
import optuna
import json
import logging
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split

"""
Changes Made:
1. Moved hyperparameters to a global dict to make tracking and changing values easier
2. Fixed some hyperparameters (batch_size specifically) being hardcoded in some places
3. batch_size value was never used
4. Revised model training to track validation & training losses and accuracies in lists
5. Implemented display_graphs to track model fitting based on those tracked values (called in train_model)
"""
# TODO: Check out Tensorboard for logging & visualization

class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 512 * 2 * 2)  # Flatten the tensor
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


hyperparameters =  {
    'batch_size': 256,
    'learning_rate': 0.00029547769932519556,
    'num_epochs': 300,
    'gamma': 0.1342360529056426,
    'step_size':30
}
# Configure logging
logging.basicConfig(
    filename="training.log",  # Log file na me
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Timestamp format
)

logging.info("\n" + "="*20)
logging.info(f"New Training Run Started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nHyperparameters: {json.dumps(hyperparameters, indent=2)}")
logging.info("="*20 + "\n")

def data_preparation(batch_size=hyperparameters["batch_size"]):
    # Enhanced Data Augmentation and Data Loading
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Normalization the data based on the mean and strd deviation values of the CIFAR-10 dataset
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load the CIFAR-10 dataset with enhanced augmentations
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                          transform=transform_train)  # train=True specifies that full_train_dataset is the training portion of CIFAR-10. The training dataset is 50,000 images.
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                    transform=transform_test)  # train=False specifies that test_dataset is the test portion of CIFAR-10. The test version of the dataset is 10,000 images.

    # Split the training data into training and validation datasets
    train_size = int(0.8 * len(full_train_dataset))  # 80% for training
    val_size = len(full_train_dataset) - train_size  # 20% for validation

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Create data loaders for training, validation, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader

# Visualization of Augmented and Unaltered Images

def objective(device, trial):
    # Suggest hyperparameters
    suggested_batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    num_epochs = trial.suggest_int("num_epochs", 10, 30)
    gamma = trial.suggest_float("gamma", 0.1, 0.9)
    step_size = trial.suggest_int("step_size", 5, 10)


    print(f"Starting Trial #{trial.number+1}")
    logging.info(f"Starting Trial #{trial.number+1}")

    # Update hyperparameter set
    hyperparameters["learning_rate"] = learning_rate
    hyperparameters["num_epochs"] = num_epochs
    hyperparameters["gamma"] = gamma
    hyperparameters["step_size"] = step_size
    hyperparameters["batch_size"] = suggested_batch_size

    logging.info(f"Trial Hyperparameters: {json.dumps(hyperparameters, indent=2)}")
    # Data preparation
    _, _, _, train_loader, val_loader, _ = data_preparation(suggested_batch_size)

    # Initialize model, loss, and optimizer
    model = ImprovedCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Training loop for objective
    best_val_accuracy = 0
    for epoch in range(hyperparameters["num_epochs"]):
        train_model(device, model, train_loader, val_loader, criterion, optimizer, scheduler, display=False)
        val_accuracy = evaluate_model(device, model, val_loader)
        best_val_accuracy = max(best_val_accuracy, val_accuracy)

        # Log intermediate results
        trial.report(val_accuracy, epoch)

        # If the trial is pruned, stop early
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_accuracy


def autotune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

    # Use lambda to pass device
    study.optimize(lambda trial: objective(device, trial), n_trials=30)

    # Print the best hyperparameters and accuracy
    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best validation accuracy: {study.best_value}")

    # Log results to file
    logging.info(f"Best hyperparameters: {study.best_params}")
    logging.info(f"Best validation accuracy: {study.best_value}")

def visualize_data(loader, title, num_images=5):
    data_iter = iter(loader)
    images, labels = next(data_iter)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    fig.suptitle(title, fontsize=16)

    # Mean and std from normalization
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])

    for i in range(num_images):
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        img = (img * std.numpy()) + mean.numpy()  # Unnormalize
        img = np.clip(img, 0, 1)  # Ensure the values are within [0, 1]
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.show()

def visualize_unaltered_images(val_dataset, test_dataset):
    # Create loaders with no augmentation for visualization
    train_loader_no_aug = DataLoader(
        datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=hyperparameters["batch_size"], shuffle=True)

    val_loader_no_aug = DataLoader(val_dataset, batch_size=hyperparameters["batch_size"], shuffle=False)
    test_loader_no_aug = DataLoader(test_dataset, batch_size=hyperparameters["batch_size"], shuffle=False)

    # Visualize unaltered training images
    visualize_data(train_loader_no_aug, "Unaltered Training Images")

    # Visualize validation images (unaltered)
    visualize_data(val_loader_no_aug, "Unaltered Validation Images")

    # Visualize test images (unaltered)
    visualize_data(test_loader_no_aug, "Unaltered Test Images")

def display_graphs(train_losses, val_losses, train_accuracies, val_accuracies):
    # Training vs Validation Loss Graph
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def train_model(device, model, train_loader, val_loader, criterion, optimizer, scheduler, display=True):
    print(json.dumps(hyperparameters, indent=2))
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(hyperparameters["num_epochs"]):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move inputs to device
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)  # Move inputs to device
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)

        logging.info(f"Epoch [{epoch + 1}/{hyperparameters['num_epochs']}], Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.2f}%, Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.2f}%")

        print(f"Epoch [{epoch + 1}/{hyperparameters['num_epochs']}], Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.2f}%, Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.2f}%")

        scheduler.step()
    # Hacky solution to prevent 50 graphs from generating during trials
    if display:
        display_graphs(train_losses, val_losses, train_accuracies, val_accuracies)
# Evaluation Function
def evaluate_model(device, model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move inputs to device
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logging.info(f"Test Accuracy of the model on the CIFAR-10 test images: {accuracy:}%")
    print(f"Test Accuracy of the model on the CIFAR-10 test images: {accuracy:.2f}%")
    return accuracy
# Function to Plot Sample Predictions
def plot_sample_predictions(device, model, test_loader, classes):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)  # Corrected this line
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)

    # Plot the first 10 test images with their predictions
    fig = plt.figure(figsize=(12, 8))
    for idx in range(10):
        ax = fig.add_subplot(2, 5, idx + 1, xticks=[], yticks=[])
        img = images[idx].cpu().numpy().transpose((1, 2, 0))
        img = np.clip((img * 0.5 + 0.5), 0, 1)  # Undo normalization for display
        ax.imshow(img)
        ax.set_title(f"{classes[predictions[idx]]}", color=("green" if predictions[idx] == labels[idx] else "red"))

    plt.show()

def main():
    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if using GPU
    if device.type == 'cuda':
        print("GPU is available and being used.")
    else:
        print("Using CPU.")

    model = ImprovedCNN().to(device)

    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])  # Optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hyperparameters["step_size"], gamma=hyperparameters["gamma"])  # Reduce LR over time

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = data_preparation()
    """
    visualize_unaltered_images(val_dataset, test_dataset)
    """
    # Train and Validate the Model
    train_model(device, model, train_loader, val_loader, criterion, optimizer, scheduler)
    # Call the export function after training the model
    evaluate_model(device, model, test_loader)

    """
    # Define class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Plot some sample predictions
    plot_sample_predictions(model, test_loader, classes)
    """

if __name__ == "__main__":
    main()
    #autotune()