import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from LeavesDataset import LeavesDataset


def main():
    train_df = pd.read_csv('classify-leaves/train.csv')
    # 获取类别列表
    classes = train_df['label'].unique()

    # 数据增强
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img_path = 'classify-leaves'
    test_path = 'classify-leaves/test.csv'
    test_dataset = LeavesDataset(test_path, img_path, mode='test', transform=data_transform)
    test_loader = DataLoader(test_dataset,batch_size=8,shuffle=False)


    # 读取网络模型
    save_path = 'weights/resnet50.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet50()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(classes))
    model.load_state_dict(torch.load(save_path))
    model.to(device)


    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            outputs = model(batch)
            preds.extend(outputs.argmax(dim=-1).cpu().numpy().tolist())

    for i in range(len(preds)):
        preds[i] = classes[preds[i]]

    test_df = pd.read_csv(test_path)
    test_df['label'] = pd.Series(preds)
    submission = pd.concat([test_df['image'], test_df['label']], axis=1)
    submission.to_csv('./submission.csv', index=False)

if __name__ == '__main__':
    main()
