# 首先导入包
import sys

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchvision.transforms
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from LeavesDataset import LeavesDataset
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models as models
# This is for the progress bar.
from tqdm import tqdm
import seaborn as sns


def main():
    train_df = pd.read_csv('classify-leaves/train.csv')
    # print(train_df.head())
    # print(train_df.info())
    # print(train_df.describe())

    classes = train_df['label'].unique()
    cls_to_idx = dict(zip(classes,range(len(classes))))
    print(cls_to_idx)

    img_path = 'classify-leaves'
    train_path = 'classify-leaves/train.csv'
    test_path = 'classify-leaves/test.csv'

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    train_dataset = LeavesDataset(train_path, img_path, mode='train', transform=data_transform['train'],
                                  cls_to_idx=cls_to_idx)
    train_len = len(train_dataset)
    valid_dataset = LeavesDataset(train_path, img_path, mode='valid', transform=data_transform['train'],
                                  cls_to_idx=cls_to_idx)
    valid_len = len(valid_dataset)
    test_dataset = LeavesDataset(test_path, img_path, mode='test', transform=data_transform['test'])
    print(f"using {train_len} data for traing. {valid_len} data for testing")

    batchsize = 4
    nw = min([os.cpu_count(), batchsize if batchsize > 1 else 0, 8])
    print(f"using {nw} dataloader workers every process")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=nw
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=nw
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=nw
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device")

    # 实例化模型,并基于迁移学习
    model = torchvision.models.resnet50(pretrained=True)

    # 冻结所有权重
    for param in model.parameters():
        param.requires_grad = False

    # 修改最后一层全连接层
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(cls_to_idx))
    model.to(device)

    # 定义损失函数，优化器
    loss_fn = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optmizer = optim.Adam(params, lr=0.0001, weight_decay=1e-4)

    epochs = 1
    best_acc = 0.0
    if not os.path.exists('./weights'):
        os.mkdir('./weights')

    save_path = './weights/resnet50.pth'
    for epoch in range(epochs):
        #----------train-----------
        model.train()
        train_pbar = tqdm(train_loader, file=sys.stdout)
        train_loss = []
        train_acc = []
        for batch in train_pbar:
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(model(imgs), labels)
            acc = (outputs.argmax(dim=-1) == labels).float().mean()
            train_loss.append(loss)
            train_acc.append(acc)

            train_pbar.desc = "train epoch [{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                       epochs,
                                                                       loss)

            # 清空梯度
            optmizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optmizer.step()

        mean_loss = sum(train_loss) / len(train_loss)
        mean_acc = sum(train_acc) / len(train_acc)
        print(f"[Train {epoch+1}/{epochs} loss: {mean_loss:.3f} acc:{mean_acc:.3f}")

        #--------valid---------
        model.eval()
        with torch.no_grad():
            valid_loss = []
            valid_acc = []
            valid_pbar = tqdm(valid_loader,file=sys.stdout)
            for batch in valid_pbar:
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                acc = (outputs.argmax(dim=-1) == labels).float().mean()
                valid_loss.append(loss)
                valid_acc.append(acc)

                train_pbar.desc = "valid epoch [{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                           epochs,
                                                                           loss)

        mean_loss = sum(valid_loss) / len(valid_loss)
        mean_acc = sum(valid_acc) / len(valid_acc)
        print(f"[Valid {epoch + 1}/{epochs} loss: {mean_loss:.3f} acc:{mean_acc:.3f}")

        if(mean_acc > best_acc):
            best_acc = mean_acc
            torch.save(model.state_dict(),save_path)

    print("finished training")

    # ---------predict--------
    model = torchvision.models.resnet50()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features,len(classes))
    model.load_state_dict(torch.load(save_path))

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            outputs = model(batch)
            preds.extend(outputs.argmax(dim=-1).cpu().numpy().tolist())

    for i in range(len(preds)):
        preds[i] = classes[preds[i]]

    test_df = pd.read_csv(test_path)
    test_df['label'] = pd.Series(preds)
    submission = pd.concat([test_df['image'],test_df['label']],axis=1)
    submission.to_csv('./submission.csv',index=False)

if __name__ == '__main__':
    main()
