import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from PIL import Image
import json
import os
from preprocessing.vocab import Vocab
import numpy as np


class VNAddressDataset(Dataset):
    def __init__(self, datadir, transform=None):
        self.data_dir = datadir
        self.transform = transform
        self.imgs = sorted(
            [n for n in os.listdir(datadir) if n[len(n) - 4:] == '.png'])
        with open(datadir + "labels.json") as file:
            self.labels = json.load(file)

    def __getitem__(self, idx):
        file_image = self.imgs[idx]
        img_path = os.path.join(self.data_dir, file_image)
        # get image
        img = Image.open(img_path).convert('L')
        # get label from image name
        label = self.labels[file_image]  # str
        target = Vocab().encode(label)  # list
        # convert class name to number
        # target = all_class.index(label)

        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)


def collate_fn(batch):
    masked_language_model = True
    filenames = []
    # img = []
    target_weights = []
    tgt_input = []
    max_label_len = max(len(sample[1]) for sample in batch)
    for sample in batch:
        # img.append(sample[0])
        label = sample[1]

        tgt = np.concatenate((label,
                              np.zeros(max_label_len - len(label),
                                       dtype=np.int32)))
        tgt_input.append(tgt)

        one_mask_len = len(label) - 1

        target_weights.append(np.concatenate((
            np.ones(one_mask_len, dtype=np.float32),
            np.zeros(max_label_len - one_mask_len, dtype=np.float32))))

    # print([i.shape for i in img])
    # img = torch.cat(img, dim=0)

    tgt_input = np.array(tgt_input, dtype=np.int64).T  # why have to transpose?
    tgt_output = np.roll(tgt_input, -1, 0).T
    tgt_output[:, -1] = 0

    # random mask token
    if masked_language_model:
        mask = np.random.random(size=tgt_input.shape) < 0.05
        mask = mask & (tgt_input != 0) & (tgt_input != 1) & (tgt_input != 2)
        tgt_input[mask] = 3

    tgt_padding_mask = np.array(target_weights) == 0
    imgs = list(zip(*batch))[0]
    imgs = torch.cat([x.unsqueeze(0) for x in imgs], dim=0)
    # print(imgs.shape)

    rs = {
        'img': imgs,
        'tgt_input': torch.LongTensor(tgt_input),
        'tgt_output': torch.LongTensor(tgt_output),
        'tgt_padding_mask': torch.BoolTensor(tgt_padding_mask),
    }
    return rs


# batch_size = 4


def create_train_test_loader(train_pre_dir, test_pre_dir, batch_size,
                             collate_function=collate_fn):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 1024)),
        transforms.Normalize(mean=0.485, std=0.229),
    ])

    train_dataset = VNAddressDataset(train_pre_dir, data_transform)
    test_dataset = VNAddressDataset(test_pre_dir, data_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=2, collate_fn=collate_function
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=2, collate_fn=collate_function
    )
    return train_loader, test_loader
