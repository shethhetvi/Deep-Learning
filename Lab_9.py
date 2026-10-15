
# PRACTICAL 1: Comparison of 7 Optimizers on CIFAR-10

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Device:", device)

# 2. Dataset and transformations
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=False,
    transform=train_transform
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=False,
    transform=test_transform
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2
)

# 3. CNN architecture
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
        x = self.features(x)
        return self.classifier(x)


# 4. Optimizer selection
def get_optimizer(name, model):
    if name == "SGD":
        return optim.SGD(model.parameters(), lr=0.01)

    elif name == "Momentum":
        return optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9
        )

    elif name == "NAG":
        return optim.SGD(
            model.parameters(), lr=0.01,
            momentum=0.9, nesterov=True
        )

    elif name == "AdaGrad":
        return optim.Adagrad(model.parameters(), lr=0.01)

    elif name == "RMSProp":
        return optim.RMSprop(
            model.parameters(), lr=0.001, alpha=0.9
        )

    elif name == "Adam":
        return optim.Adam(
            model.parameters(), lr=0.001,
            weight_decay=1e-4
        )

    elif name == "AdamW":
        return optim.AdamW(
            model.parameters(), lr=0.001,
            weight_decay=1e-4
        )


# 5. Train and evaluate one model
def train_model(model, optimizer, epochs=10):
    criterion = nn.CrossEntropyLoss()
    target_layer = model.classifier[4]

    losses = []
    accuracies = []
    weight_norms = [
        torch.norm(target_layer.weight.detach()).item()
    ]
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

        # Test accuracy
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                predictions = outputs.argmax(dim=1)

                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        accuracy = 100 * correct / total
        accuracies.append(accuracy)

        # Weight magnitude and update size
        current_weights = target_layer.weight.detach().clone()

        weight_norms.append(
            torch.norm(current_weights).item()
        )
        delta_norms.append(
            torch.norm(current_weights - previous_weights).item()
        )

        previous_weights = current_weights.clone()

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

    return losses, accuracies, weight_norms, delta_norms


# 6. Run all seven optimizers
optimizer_names = [
    "SGD", "Momentum", "NAG", "AdaGrad",
    "RMSProp", "Adam", "AdamW"
]

history = {}

for name in optimizer_names:
    print("\nRunning optimizer:", name)

    # Fresh model for a fair comparison
    model = CIFAR10ConvNet().to(device)
    optimizer = get_optimizer(name, model)

    history[name] = train_model(model, optimizer, epochs=10)


# 7. Plot the four comparison graphs
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for name in optimizer_names:
    axes[0, 0].plot(
        history[name][0], marker="o", label=name
    )
axes[0, 0].set_title("Training Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Cross-Entropy Loss")
axes[0, 0].legend()
axes[0, 0].grid(True)

for name in optimizer_names:
    axes[0, 1].plot(
        history[name][1], marker="s", label=name
    )
axes[0, 1].set_title("Test Accuracy")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy (%)")
axes[0, 1].legend()
axes[0, 1].grid(True)

for name in optimizer_names:
    axes[1, 0].plot(
        history[name][2], marker="^", label=name
    )
axes[1, 0].set_title("Weight Magnitude (L2 Norm)")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Weight Norm")
axes[1, 0].legend()
axes[1, 0].grid(True)

for name in optimizer_names:
    axes[1, 1].plot(
        history[name][3], marker="d", label=name
    )
axes[1, 1].set_title("Weight Update Size")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Delta Weight Norm")
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig("cifar10_optimizer_results.png", dpi=300)
plt.show()

print("Results saved to cifar10_optimizer_results.png")
