import math
import time
import os
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import models

from dataloader import StanfordDogsDataset


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

dataloaders = {
    "train": DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4),
    "validation": DataLoader(validation_set, batch_size=16, shuffle=True, num_workers=4),
}
dataset_sizes = {
    "train": len(train_set),
    "validation": len(validation_set),
}


model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 120)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

for epoch in count():
    print(f"Epoch {epoch + 1}")
    print("-" * 10)
    since = time.thread_time()

    # Each epoch has a training and validation phase
    for phase in ["train", "validation"]:
        if phase == "train":
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase] * 100

        time_elapsed = time.thread_time() - since
        print(
            "{} loss: {:.4f}, accuracy: {:.4f}%, time: {:.0f}m {:.0f}s".format(
                phase, epoch_loss, epoch_acc, time_elapsed // 60, time_elapsed % 60
            )
        )
        if phase == "train":
            torch.save(model.state_dict(), SAVE_FILE)
    print()
