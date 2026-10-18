
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Device:", device)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=train_transform)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=test_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

class CIFAR10ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def get_optimizer(name, model):
    if name == "SGD":
        return optim.SGD(model.parameters(), lr=0.01)
    if name == "Momentum":
        return optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    if name == "NAG":
        return optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    if name == "AdaGrad":
        return optim.Adagrad(model.parameters(), lr=0.01)
    if name == "RMSProp":
        return optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
    if name == "Adam":
        return optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    if name == "AdamW":
        return optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    raise ValueError("Unknown optimizer: " + name)

def train_model(model, optimizer, epochs=10):
    criterion = nn.CrossEntropyLoss()
    target_layer = model.classifier[4]
    losses = []
    accuracies = []
    weight_norms = [torch.norm(target_layer.weight.detach()).item()]
    delta_norms = []
    previous_weights = target_layer.weight.detach().clone()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(trainloader.dataset)
        losses.append(epoch_loss)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = 100.0 * correct / total
        accuracies.append(accuracy)

        current_weights = target_layer.weight.detach().clone()
        weight_norm = torch.norm(current_weights).item()
        delta_norm = torch.norm(current_weights - previous_weights).item()
        weight_norms.append(weight_norm)
        delta_norms.append(delta_norm)
        previous_weights = current_weights

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}% | "
            f"Weight Norm: {weight_norm:.6f} | "
            f"Step Size: {delta_norm:.6f}"
        )

    return losses, accuracies, weight_norms, delta_norms

def main():
    optimizer_names = ["SGD", "Momentum", "NAG", "AdaGrad", "RMSProp", "Adam", "AdamW"]
    history = {}

    for index, name in enumerate(optimizer_names):
        print("\nRunning optimizer:", name)
        torch.manual_seed(100 + index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(100 + index)

        model = CIFAR10ConvNet().to(device)
        optimizer = get_optimizer(name, model)
        history[name] = train_model(model, optimizer, epochs=10)

        loss, accuracy, weight_norm, delta_norm = history[name]
        print("\nFINAL RESULTS:", name)
        print(f"Final Training Loss: {loss[-1]:.4f}")
        print(f"Final Test Accuracy: {accuracy[-1]:.2f}%")
        print(f"Final Weight Norm: {weight_norm[-1]:.6f}")
        print(f"Final Step Size: {delta_norm[-1]:.6f}")
        print("-" * 55)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for name in optimizer_names:
        loss, accuracy, weight_norm, delta_norm = history[name]
        axes[0, 0].plot(range(1, 11), loss, marker="o", label=name)
        axes[0, 1].plot(range(1, 11), accuracy, marker="s", label=name)
        axes[1, 0].plot(range(0, 11), weight_norm, marker="^", label=name)
        axes[1, 1].plot(range(1, 11), delta_norm, marker="d", label=name)

    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Cross-Entropy Loss")

    axes[0, 1].set_title("Test Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")

    axes[1, 0].set_title("Weight Magnitude (L2 Norm)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Weight Norm")

    axes[1, 1].set_title("Weight Update Size")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Delta Weight Norm")

    for ax in axes.flat:
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("cifar10_optimizer_results.png", dpi=300)
    plt.show()

    print("\nCOMPLETE COMPARISON TABLE")
    print(f"{'Optimizer':<12}{'Loss':>12}{'Accuracy':>14}{'Weight Norm':>16}{'Step Size':>14}")

    for name in optimizer_names:
        loss, accuracy, weight_norm, delta_norm = history[name]
        print(
            f"{name:<12}{loss[-1]:>12.4f}"
            f"{accuracy[-1]:>13.2f}%"
            f"{weight_norm[-1]:>16.6f}"
            f"{delta_norm[-1]:>14.6f}"
        )

    print("\nGraph saved as cifar10_optimizer_results.png")

if __name__ == "__main__":
    main()
