from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import pandas as pd
import numpy as np

import zipfile
from urllib import request


def download_ml100k():
    # download
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    savename = "ml-100k.zip"
    request.urlretrieve(url, savename)
    print('저장되었습니다')
    # unzip
    file_name = os.path.join('./', savename)
    file_zip = zipfile.ZipFile(file_name)
    file_zip.extractall('./')
    file_zip.close()


def read_data_ml100k():
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join('./ml-100k/', 'u.data'), '\t', names=names,
                       engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items


def load_data_ml100k(data, num_users, num_items, feedback='explicit'):
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter


def split_data_ml100k(data, num_users, num_items,
                      split_mode='random', test_ratio=0.1):
    """Split the dataset in random mode or seq-aware mode."""
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data


class ArrayDataset(Dataset):

    def __init__(self, user, item, rating, transform=None):
        self.user = user
        self.item = item
        self.rating = rating
        self.transform = transform

    def __getitem__(self, idx):
        arr = np.column_stack((self.user, self.item, self.rating))
        if self.transform:
            arr = self.transform(arr)
        return arr[idx]

    def __len__(self):
        return len(self.user)


def collate_batch(batch):
    batch = np.array(batch)
    return batch[:, 0], batch[:, 1], batch[:, 2]

# @save


def split_and_load_ml100k(split_mode='seq-aware', feedback='explicit',
                          test_ratio=0.1, batch_size=256):
    data, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode, test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback)
    train_set = ArrayDataset(
        np.array(train_u), np.array(train_i), np.array(train_r))
    test_set = ArrayDataset(
        np.array(test_u), np.array(test_i), np.array(test_r))
    train_iter = DataLoader(
        train_set, shuffle=True,
        batch_size=batch_size, collate_fn=collate_batch)
    test_iter = DataLoader(
        test_set, batch_size=batch_size, collate_fn=collate_batch)
    return num_users, num_items, train_iter, test_iter


if not os.path.exists('ml-100k'):
    # 파일 다운받을 때 한번만 실행
    download_ml100k()
if __name__ == "__main__":
  num_users, num_items, train_iter, test_iter = split_and_load_ml100k()
