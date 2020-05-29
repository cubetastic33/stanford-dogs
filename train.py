from itertools import count
import os
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt

from dataloader import StanfordDogsDataset


class HAL9000(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.maxpool4 = nn.MaxPool2d(2, 2)
        self.bn4 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=120)
        self.dropout = nn.Dropout(.5)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.bn1(self.maxpool1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.bn2(self.maxpool2(x))
        x = F.leaky_relu(self.conv3(x))
        x = self.bn3(self.maxpool3(x))
        x = F.leaky_relu(self.conv4(x))
        x = self.bn3(self.maxpool4(x))
        x = F.leaky_relu(self.dropout(self.fc1(x.view(-1, 256))))
        x = F.softmax(self.fc2(x), dim=1)
        return x


def preprocess(image):
    width, height = image.size
    if width > height and width > 512:
        height = math.floor(512 * height / width)
        width = 512
    elif width < height and height > 512:
        width = math.floor(512 * width / height)
        height = 512
    pad_values = (
        (512 - width) // 2 + (0 if width % 2 == 0 else 1),
        (512 - height) // 2 + (0 if height % 2 == 0 else 1),
        (512 - width) // 2,
        (512 - height) // 2,
    )
    return T.Compose([
        T.RandomGrayscale(),
        T.Resize((height, width)),
        T.Pad(pad_values),
        T.ToTensor(),
        T.Lambda(lambda x: x[:3]),  # Remove the alpha channel if it's there
    ])(image)


DEVICE = torch.device("cuda")
SAVE_FILE = "HAL9000.pt"

train_set = StanfordDogsDataset(
    root=os.path.join(os.getcwd(), "data"), set_type="train", transform=preprocess
)
validation_set = StanfordDogsDataset(
    root=os.path.join(os.getcwd(), "data"), set_type="validation", transform=preprocess
)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)
validation_loader = DataLoader(validation_set, batch_size=16, shuffle=True, num_workers=4)
network = HAL9000().to(DEVICE)
optimizer = optim.Adam(network.parameters(), lr=.001)
epoch = 0
losses = []

if os.path.isfile(SAVE_FILE):
    checkpoint = torch.load(SAVE_FILE)
    epoch = checkpoint["epoch"]
    network.load_state_dict(checkpoint["network"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def validate():
    total_correct = 0
    validation_losses = []
    for images, labels in validation_loader:
        with torch.no_grad():
            # Get the predictions
            predictions = network(images.to(DEVICE))
            # Calculate the loss
            loss = F.cross_entropy(predictions, labels.to(DEVICE))

            num_correct = predictions.argmax(dim=1).eq(labels.to(DEVICE)).sum().item()
            total_correct += num_correct
            validation_losses.append(loss.item())

    print(
        "Validation accuracy: "
        + str(round(total_correct / len(validation_set) * 100, 2))
        + "%",
        f"Average loss in validation: {round(torch.tensor(validation_losses, device=DEVICE).mean().item(), 4)}",
        sep=", "
    )


for i in count(epoch):
    total_correct = 0
    start_time = time.thread_time()

    for images, labels in train_loader:
        # Get the predictions
        predictions = network(images.to(DEVICE))
        # Calculate the loss
        loss = F.cross_entropy(predictions, labels.to(DEVICE))

        # Reset the gradients
        optimizer.zero_grad()
        # Calculate the gradients
        loss.backward()
        # Update the network
        optimizer.step()

        num_correct = predictions.argmax(dim=1).eq(labels.to(DEVICE)).sum().item()
        total_correct += num_correct
        losses.append(loss.item())

    print(
        f"Epoch: {i + 1}",
        "Accuracy: "
        + str(round(total_correct / len(train_set) * 100, 2))
        + "%",
        f"Average loss: {round(torch.tensor(losses[-len(train_loader):], device=DEVICE).mean().item(), 4)}",
        f"Time: {round(time.thread_time() - start_time, 2)}",
        sep=", "
    )
    checkpoint = {
        "epoch": i + 1,
        "network": network.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, SAVE_FILE)
    # Uncomment for loss graph
    # plt.ion()
    # plt.title("Loss")
    # plt.plot(losses)
    # plt.ioff()
    # plt.show()
    validate()
    print()
