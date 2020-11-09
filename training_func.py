import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

def train_model(model: nn.Module,
          iterator: DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    fr = 0
    fa = 0
    count = 1
    for input, labels in train_dataloader:
#         print(count)
        count += 1
        input = input.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        input = input.permute(0, 2, 1)
        output = model(input)
        loss = criterion(output, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        _, preds = torch.max(output, 1)
        epoch_loss += loss.item() * input.shape[0]
        epoch_accuracy += (preds == labels).sum().item()
        for i in range(labels.shape[0]):
            if preds[i] == 0 and labels[i] == 1:
                fr += 1
            elif preds[i] == 1 and labels[i] == 0:
                fa += 1
    count_all = len(train_dataloader) * 128
    epoch_accuracy = epoch_accuracy / count_all
    epoch_loss = epoch_loss / count_all
    return epoch_loss, epoch_accuracy, fr / count_all, fa / count_all


def evaluate(model: nn.Module,
             iterator: DataLoader,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0
    epoch_accuracy = 0
    roc_auc = 0
    fr = 0
    fa = 0
    count = 1
    with torch.no_grad():

        for input, labels in test_dataloader:
#             print(count)
            count += 1
            input = input.to(device)
            labels = labels.to(device)
            input = input.permute(0, 2, 1)
            output = model(input)
            _, preds = torch.max(output, 1)

            loss = criterion(output, labels)

            epoch_loss += loss.item() * input.size(0)
            epoch_accuracy += (preds == labels).sum().item()
            for i in range(labels.shape[0]):
                if preds[i] == 0 and labels[i] == 1:
                    fr += 1
                elif preds[i] == 1 and labels[i] == 0:
                    fa += 1
    count_all = len(test_dataloader) * 128
    epoch_accuracy = epoch_accuracy / count_all
    epoch_loss = epoch_loss / count_all
    return epoch_loss, epoch_accuracy, fr / count_all, fa / count_all
