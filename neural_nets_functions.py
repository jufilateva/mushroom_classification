import torch
from datetime import datetime


# функция обучения нейросети (одна эпоха)
def train_epoch(model, device, dataloader, loss_fn, optimizer):
    train_loss, train_correct = 0.0, 0
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()

    return train_loss, train_correct


# функция валидации нейросети (одна эпоха)
def valid_epoch(model, device, dataloader, loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        output.tolist()
        loss = loss_fn(output, labels)
        valid_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        print(output.data)
        val_correct += (predictions == labels).sum().item()

    return valid_loss, val_correct


# функция тестирования обеих нейронных сетей. Подсчет правильно распознанных пар
def test(model1, model2, device, dataloader1, dataloader2):
    test_correct = 0
    model1.eval()
    model2.eval()
    i = 0

    # создание массива, куда для каждого изображения запишется
    # вероятность его принадлежности к каждому класу
    sum_list = []
    for images1, labels1 in dataloader1:
        images1, labels1 = images1.to(device), labels1.to(device)
        output1 = model1(images1)
        output1 = output1.tolist()
        # добавляем в список массив вероятностей для каждого изображения
        sum_list.append(output1)

    for images2, labels2 in dataloader2:
        images2, labels2 = images2.to(device), labels2.to(device)
        output2 = model2(images2)
        output2 = output2.tolist()
        for j in range(len(sum_list[i])):
            for k in range(len(sum_list[i][j])):
                # теперь каждый элемент массива заполняем усредненным значением
                # вероятностей для обоих ракурсов
                sum_list[i][j][k] = (
                    float(sum_list[i][j][k]) * 0.5 + float(output2[j][k]) * 0.5
                )
        predictions = sum_list[i][j].index(max(sum_list[i][j]))
        # считаем количество правильно предсказанных пар
        test_correct += (predictions == labels2).sum().item()
        i += 1
    return test_correct


# функция обучения нейросети для набора данных "Сверху"
def train_up(
    transfer_model_up, dataloader_up, criterion, optimizer, num_epochs, splits, sum_up
):
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(sum_up))):
        print("Fold {}".format(fold + 1))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = transfer_model_up

        # для каждого блока проходим заявленное количество эпох
        for epoch in range(num_epochs):
            start = datetime.now()

            train_loss, train_correct = train_epoch(
                model, device, dataloader_up["train"], criterion, optimizer
            )

            val_loss, val_correct = valid_epoch(
                model, device, dataloader_up["val"], criterion
            )

            train_loss = train_loss / len(dataloader_up["train"].sampler)
            train_acc = train_correct / len(dataloader_up["train"].sampler) * 100
            val_loss = val_loss / len(dataloader_up["val"].sampler)
            val_acc = val_correct / len(dataloader_up["val"].sampler) * 100

            print(
                "Epoch {}/{} Training Loss:{:.3f},  Val Loss:{:.3f},  Training Acc {:.2f} %,  Val Acc {:.2f} %".format(
                    epoch + 1, num_epochs, train_loss, val_loss, train_acc, val_acc
                )
            )
            print(datetime.now() - start)

            # сохранение модели
            model_path = "/content/drive/MyDrive/verh_model_2.pt"
            torch.save(model, model_path)


# функция обучени нейросети для набора данных "Сбоку"
def train_slide(
    transfer_model_slide,
    dataloader_slide,
    criterion,
    optimizer,
    num_epochs,
    splits,
    sum_slide,
):
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(sum_slide))):
        print("Fold {}".format(fold + 1))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = transfer_model_slide

        history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

        for epoch in range(num_epochs):
            start = datetime.now()

            train_loss, train_correct = train_epoch(
                model, device, dataloader_slide["train"], criterion, optimizer
            )

            val_loss, val_correct = valid_epoch(
                model, device, dataloader_slide["val"], criterion
            )

            train_loss = train_loss / len(dataloader_slide["train"].sampler)
            train_acc = train_correct / len(dataloader_slide["train"].sampler) * 100
            val_loss = val_loss / len(dataloader_slide["val"].sampler)
            val_acc = val_correct / len(dataloader_slide["val"].sampler) * 100

            print(
                "Epoch {}/{} Training Loss:{:.3f},  Val Loss:{:.3f},  Training Acc {:.2f} %,  Val Acc {:.2f} %".format(
                    epoch + 1, num_epochs, train_loss, val_loss, train_acc, val_acc
                )
            )
            print(datetime.now() - start)

            model_path = "/content/drive/MyDrive/bok_model_2.pt"
            torch.save(model, model_path)
