from torchvision import models
import torch
import tqdm
import os
import copy

from data.dog_breed_identification import dog_classfication_dataset as data
import utils

ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 120
EPOCHS = 20
BATCH_SIZE = 32


def buile_model():
    """
    构造模型
    :param num_classes: 分类📚
    :return:
    """
    model = models.resnet50(weights=True)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def train(model, optimizer, criterion, scheduler, train_loader, val_loader, save_path):
    timer = utils.Timer()
    timer.start()

    best_acc = 0
    model.to(DEVICE)

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    val_losses = []
    LRs = [optimizer.param_groups[0]['lr']]

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = tqdm.tqdm(train_loader)
            else:
                model.eval()
                data_loader = tqdm.tqdm(val_loader)

            run_loss = 0.0
            run_corrects = 0

            for inputs, labels in data_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                run_loss += loss.item() * inputs.size(0)
                run_corrects += torch.sum(torch.tensor(preds == labels.data))

            epoch_loss = run_loss / len(data_loader)
            epoch_acc = run_corrects.double() / len(data_loader)

            timer.end()

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {'state_dict': model.state_dict(), 'best_acc': best_acc, 'optimizer': optimizer.state_dict()}
                torch.save(state, save_path)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
        print('优化器学习率:{:.7f'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    print("训练完成")
    timer.end()
    print(f"Best val Acc: {best_acc:4f}")


def main():
    print(f'在{DEVICE}上训练')
    model = buile_model()
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()
    train_loader, test_loader = data.get_dataloader(BATCH_SIZE)
    save_path = os.path.join(ROOT, "dog_classification.pth")
    train(model, optimizer, criterion, scheduler, train_loader, test_loader, save_path)


if __name__ == '__main__':
    main()
