import torch
from torch import nn
import torch.optim as optim
import torchvision
import tqdm
import os
import copy
from torch.utils.data import DataLoader

from data.dog_breed_identification import dog_classfication_dataset as data
import utils

ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 120
EPOCHS = 50
BATCH_SIZE = 128


def buile_model():
    """
    构造模型
    :param num_classes: 分类📚
    :return:
    """
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def train(model, optimizer, criterion, scheduler, data_package, save_path):
    timer = utils.Timer()
    timer.start()
    best_acc = 0
    model.to(DEVICE)

    train_loader = DataLoader(data_package[0], batch_size=BATCH_SIZE)
    val_loader = DataLoader(data_package[1], batch_size=BATCH_SIZE)

    data_map = {"train": train_loader, "valid": val_loader}

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch, EPOCHS - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 验证

            data_loader = data_map[phase]
            data_num = len(data_package[0]) if phase == 'train' else len(data_package[1])

            running_loss = 0.0
            running_corrects = 0

            # 把数据都取个遍
            for inputs, labels in tqdm.tqdm(data_loader):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # 清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    # print(loss)

                    # 训练阶段更新权重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 计算损失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(torch.tensor(preds == labels.data))
            epoch_loss = running_loss / data_num
            epoch_acc = running_corrects / data_num

            timer.end()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 得到最好那次的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),  # 字典里key就是各层的名字，值就是训练好的权重
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),  # 优化器的状态信息
                }
                torch.save(state, save_path)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)  # 学习率衰减
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    timer.end()
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果,等着一会测试
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


def main():
    print(f'在{DEVICE}上训练')
    model = buile_model()
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    train_dataset, valid_dataset = data.get_dataset()
    data_package = (train_dataset, valid_dataset)
    save_path = os.path.join(ROOT, "dog_classification.pth")
    train(model, optimizer, criterion, scheduler, data_package, save_path)


if __name__ == '__main__':
    main()
