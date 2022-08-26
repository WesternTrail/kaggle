import os.path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class LeavesDataset(Dataset):
    def __init__(self, csv_path, img_path, mode=None, valid_ratio=0.2, cls_to_idx=None, transform=None):
        self.csv_path = csv_path
        self.img_path = img_path
        self.cls_to_idx = cls_to_idx
        self.transform = transform
        self.data = pd.read_csv(csv_path)
        self.mode = mode
        self.train_len = int(len(self.data) * (1 - valid_ratio))

        if self.mode == 'train':
            train_img = np.asarray(self.data.iloc[:self.train_len, 0])
            train_label = np.asarray(self.data.iloc[:self.train_len, 1])

            self.img_arr = train_img
            self.lable_arr = train_label
        elif mode == 'valid':
            valid_img = np.asarray(self.data.iloc[self.train_len:, 0])
            valid_label = np.asarray(self.data.iloc[self.train_len:, 1])
            self.img_arr = valid_img
            self.lable_arr = valid_label
        else:
            test_img = np.asarray(self.data.iloc[0:, 0])
            self.img_arr = test_img
        self.data_len = len(self.img_arr)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_path, self.img_arr[item])
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.mode == 'test':
            return img
        else:
            label_cls = self.lable_arr[item]
            label_idx = self.cls_to_idx[label_cls]
            return img, label_idx

    def __len__(self):
        return self.data_len

# if __name__ == '__main__':
#     img_path = 'classify-leaves'
#     train_path = 'classify-leaves/train.csv'
#     test_path = 'classify-leaves/test.csv'
#
#     train_dataset = LeavesDataset2(train_path,img_path,mode='train')
#     for data in train_dataset:
#         img,label = data
#         img.show()
#         print(label)
