import torch.nn as nn
from PIL import Image
from pathlib import Path
import numpy as np
import torch
import torchvision
from torchvision import models, transforms, datasets
import os
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from google.colab import drive
from sklearn.model_selection import KFold

from neural_nets_functions import train_epoch, \
                                  valid_epoch,  \
                                  test, \
                                  train_up, \
                                  train_slide

# подключение к гугл диску, где хранятся датасеты и загруженные модели
drive.mount("/content/drive", force_remount=True)

# предподготовка данных, преобразование в тензорный формат для работы с нейросетями
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}


# подготовка данных, образование наборов данных

path_up = "/content/drive/MyDrive/Сверху"
path_slide = "/content/drive/MyDrive/training/Сбоку"

# набор данных "Сверху", тренировочный и валидационный
image_datasets_up = {
    x: datasets.ImageFolder(os.path.join(path_up, x), data_transforms[x])
    for x in ["train", "val"]
}
dataloader_up = {
    x: torch.utils.data.DataLoader(
        image_datasets_up[x], batch_size=128, shuffle=True, num_workers=0
    )
    for x in ["train", "val"]
}
dataset_sizes = {x: len(image_datasets_up[x]) for x in ["train", "val"]}
class_names = image_datasets_up["train"].classes
sum_up = 0

for i in class_names:
    sum_up = sum_up + len(os.listdir(path_up + "/train/"))
    sum_up = sum_up + len(os.listdir(path_up + "/val/"))

# набор данных "Сбоку", тренировочный и валидационный
image_datasets_slide = {
    x: datasets.ImageFolder(os.path.join(path_slide, x), data_transforms[x])
    for x in ["train", "val"]
}
dataloader_slide = {
    x: torch.utils.data.DataLoader(
        image_datasets_slide[x], batch_size=128, shuffle=True, num_workers=0
    )
    for x in ["train", "val"]
}
dataset_sizes = {x: len(image_datasets_slide[x]) for x in ["train", "val"]}
class_names = image_datasets_slide["train"].classes

sum_slide = 0
for i in class_names:
    sum_slide = sum_slide + len(os.listdir(path_slide + "/train/"))
    sum_slide = sum_slide + len(os.listdir(path_slide + "/val/"))

# тетовые наборы данных
test_set_up = datasets.ImageFolder(path_up + "/test", data_transforms["val"])
test_loader_up = torch.utils.data.DataLoader(
    test_set_up, batch_size=1, shuffle=True, num_workers=0
)

test_set_slide = datasets.ImageFolder(path_slide + "/test", data_transforms["val"])
test_loader_slide = torch.utils.data.DataLoader(
    test_set_slide, batch_size=1, shuffle=True, num_workers=0
)


# загрузка модели для набора данных "Сверху"

transfer_model_up = models.resnet18(pretrained=True)
# заморозка параметров
for name, param in transfer_model_up.named_parameters():
    if "bn" not in name:
        param.requires_grad = False

# изменение полносвязных слоев под свои классы
transfer_model_up.fc = nn.Sequential(
    nn.Linear(transfer_model_up.fc.in_features, 300),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(300, 16),
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transfer_model_up = transfer_model_up.to(device)

# оптимизатор и функция потерь
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(transfer_model_up.parameters(), lr=0.002)


# загрузка модели для набора данных "Сбоку"

transfer_model_slide = models.resnet50(pretrained=True)
# заморозка параметров
for name, param in transfer_model_slide.named_parameters():
    if "bn" not in name:
        param.requires_grad = False

# изменение полносвязных слоев под свои классы
transfer_model_slide.fc = nn.Sequential(
    nn.Linear(transfer_model_slide.fc.in_features, 600),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(600, 16),
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transfer_model_slide = transfer_model_slide.to(device)

# оптимизатор и функция потерь
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transfer_model_slide.parameters(), lr=0.0002)


num_epochs = 7  # количество эпох на один блок кросс-валидации
batch_size = 128
k = 5  # количетво блоков кросс-валидации
splits = KFold(n_splits=k, shuffle=True, random_state=42)
foldperf = {}


# обучение обеих нейросетей
train_up(
    transfer_model_up,
    dataloader_up,
    criterion,
    optimizer,
    num_epochs,
    splits,
    sum_up
)
train_slide(
    transfer_model_slide,
    dataloader_slide,
    criterion,
    optimizer,
    num_epochs,
    splits,
    sum_slide,
)


# скачивание обученных моделей
model1 = torch.load("/content/drive/MyDrive/verh_model_2.pt")
model2 = torch.load("/content/drive/MyDrive/bok_model_2.pt")

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# подсчет корректно распознанных пар изображений и точности (accuracy)
test_correct = test(model1, model2, device, test_loader_up, test_loader_slide)
test_acc = test_correct / len(test_loader_up.sampler) * 100

print("Test Accuracy = {:.2f} %".format(test_acc))
