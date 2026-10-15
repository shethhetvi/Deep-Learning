import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from torch.utils.data import DataLoader

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 0.001

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("Device:", DEVICE)

# --------------------------------------------------
# 2. DEFINE LENET-5 ARCHITECTURE
# --------------------------------------------------

class LeNet5(nn.Module):
    def __init__(self, activation="tanh", pooling="avg"):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = (
            nn.AvgPool2d(2, 2) if pooling == "avg"
            else nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = (
            nn.AvgPool2d(2, 2) if pooling == "avg"
            else nn.MaxPool2d(2, 2)
        )

        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.activation = (
            nn.Tanh() if activation == "tanh" else nn.ReLU()
        )

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool1(x)

        x = self.activation(self.conv2(x))
        x = self.pool2(x)

        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        return x


# --------------------------------------------------
# 3. PRINT TENSOR SHAPES
# --------------------------------------------------

model = LeNet5().to(DEVICE)
x = torch.randn(1, 1, 32, 32).to(DEVICE)

print("\n--- Tensor Shape Verification ---")

with torch.no_grad():
    print("Input:", x.shape)

    x = torch.tanh(model.conv1(x))
    print("After Conv1:", x.shape)

    x = model.pool1(x)
    print("After Pool1:", x.shape)

    x = torch.tanh(model.conv2(x))
    print("After Conv2:", x.shape)

    x = model.pool2(x)
    print("After Pool2:", x.shape)

    x = torch.flatten(x, 1)
    print("After Flatten:", x.shape)

    x = torch.tanh(model.fc1(x))
    print("After FC1:", x.shape)

    x = torch.tanh(model.fc2(x))
    print("After FC2:", x.shape)

    x = model.fc3(x)
    print("After FC3:", x.shape)


# --------------------------------------------------
# 4. CONV1 TRAINABLE PARAMETER CALCULATION
# --------------------------------------------------

weights = model.conv1.weight.numel()
biases = model.conv1.bias.numel()

print("\n--- Conv1 Parameters ---")
print("Weights:", weights)
print("Biases:", biases)
print("Total parameters:", weights + biases)


# --------------------------------------------------
# 5. LOAD MNIST DATASET
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

print("\nTraining images:", len(train_dataset))
print("Testing images:", len(test_dataset))


# --------------------------------------------------
# 6. TRAIN AND EVALUATE THE MODEL
# --------------------------------------------------

def train_model(model, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    losses = []

    print(f"\n--- Training {model_name} ---")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}, "
            f"Loss: {epoch_loss:.4f}"
        )

    # Evaluate on test data
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Test accuracy: {accuracy:.2f}%")

    # Save trained weights
    path = f"models/{model_name}.pth"
    torch.save(model.state_dict(), path)
    print("Saved:", path)

    # Plot training loss
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"Training Loss - {model_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return accuracy


# --------------------------------------------------
# 7. TRAIN ORIGINAL AND MODIFIED MODELS
# --------------------------------------------------

original_model = LeNet5(
    activation="tanh", pooling="avg"
).to(DEVICE)

modified_model = LeNet5(
    activation="relu", pooling="max"
).to(DEVICE)

original_accuracy = train_model(
    original_model, "lenet5_original"
)

modified_accuracy = train_model(
    modified_model, "lenet5_modified"
)

print("\n--- Comparative Analysis ---")
print(f"Original accuracy: {original_accuracy:.2f}%")
print(f"Modified accuracy: {modified_accuracy:.2f}%")


# --------------------------------------------------
# 8. LOAD TRAINED WEIGHTS FOR GUI
# --------------------------------------------------

original_model.load_state_dict(
    torch.load(
        "models/lenet5_original.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

modified_model.load_state_dict(
    torch.load(
        "models/lenet5_modified.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

original_model.eval()
modified_model.eval()

models = {
    "Original: Tanh + Average Pooling": original_model,
    "Modified: ReLU + Max Pooling": modified_model
}


# --------------------------------------------------
# 9. IMAGE PREPROCESSING AND PREDICTION
# --------------------------------------------------

def predict_digit(canvas_input, model_choice):
    if canvas_input is None:
        return {"Draw a digit first": 1.0}

    if isinstance(canvas_input, dict):
        img = canvas_input.get("composite")

        if img is None:
            img = canvas_input.get("background")
    else:
        img = canvas_input

    if img is None:
        return {"Draw a digit first": 1.0}

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype("uint8"))

    if not isinstance(img, Image.Image):
        return {"Invalid image": 1.0}

    # Convert black drawing on white canvas
    # to white digit on black background
    img = ImageOps.invert(img.convert("L"))
    img = img.resize((32, 32))

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    tensor_img = preprocess(img).unsqueeze(0).to(DEVICE)

    selected_model = models[model_choice]
    selected_model.eval()

    with torch.no_grad():
        outputs = selected_model(tensor_img)
        probabilities = torch.softmax(outputs[0], dim=0)

    return {
        str(i): float(probabilities[i])
        for i in range(10)
    }


# --------------------------------------------------
# 10. BUILD GRADIO DIGIT RECOGNIZER
# --------------------------------------------------

demo = gr.Interface(
    fn=predict_digit,
    inputs=[
        gr.ImageEditor(
            type="pil",
            image_mode="RGB",
            label="Draw a digit (0-9)",
            brush=gr.Brush(
                colors=["#000000"],
                default_size=20
            )
        ),
        gr.Dropdown(
            choices=list(models.keys()),
            value="Original: Tanh + Average Pooling",
            label="Select CNN Architecture"
        )
    ],
    outputs=gr.Label(
        num_top_classes=3,
        label="Digit Predictions"
    ),
    title="LeNet-5 Handwritten Digit Recognizer",
    description=(
        "Draw a digit from 0 to 9 and submit it. "
        "Compare the original and modified CNN models."
    )
)

if __name__ == "__main__":
    demo.launch()