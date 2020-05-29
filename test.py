import math
import os

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
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
LOAD_FILE = "HAL9000.pt"

test_set = StanfordDogsDataset(
    root=os.path.join(os.getcwd(), "data"), set_type="test", transform=preprocess
)
test_loader = DataLoader(test_set, batch_size=16, shuffle=True, num_workers=4)

model = models.resnet50(pretrained=True)

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 120)
model.to(DEVICE)
model.load_state_dict(torch.load(LOAD_FILE))
model.eval()

num_correct = 0
total_loss = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        predictions = model(inputs.to(DEVICE))
        loss = F.cross_entropy(predictions, labels.to(DEVICE))
        num_correct += predictions.argmax(dim=1).eq(labels.to(DEVICE)).sum().item()
        total_loss += loss.item()

print(f"Accuracy: {num_correct / len(test_set) * 100:.2f}%, Average loss: {total_loss / len(test_set):.4f}")
