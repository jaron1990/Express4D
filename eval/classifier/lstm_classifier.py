import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
import matplotlib.pyplot as plt

import numpy as np

data_list = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
labels_map = {}
for i in range(6):
    labels_map[i] = data_list[i]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-6


class FaceData(Dataset):
    def __init__(self, data, label):
        self.pts = data
        self.labels = label
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pc = self.pts[idx]

        #   pc = np.reshape(self.pts[idx], (100, 83,3))  # 100x249 => 100x83x3
        # pc, coeff = on_unit_cube(pc)
        pc = np.reshape(pc, (100, 249))
        pc = torch.FloatTensor(pc)
        # s = torch.FloatTensor([coeff['scale']])
        # m = torch.FloatTensor(coeff['min'])
        y = self.labels[idx]
        return pc, y


class LSTMClassifier(nn.Module):

    def __init__(self, num_pts, num_frame, label_size):
        super(LSTMClassifier, self).__init__()
        # self.encoding_conv = nn.Linear(num_pts, 16)
        self.lstm_1 = nn.LSTM(249, 128, batch_first=True, bidirectional=True)
        #  self.lstm_2 = nn.LSTM(128, 128, batch_first=True, bidirectional=False)

        # self.w = nn.Parameter(torch.zeros(1024))
        #  self.linear = nn.Linear(128*2, 6)
        self.cls = nn.Linear(128 * 2, label_size)
        self.tanh1 = nn.Tanh()

    def forward(self, x):
        bs = x.shape[0]

        #  x = F.relu(self.encoding_conv(x))

        x, (hidden, cell) = self.lstm_1(x)

        # m = self.tanh1(lstm_out)
        #  alpha = F.softmax(torch.matmul(m, self.w), dim=1).unsqueeze(-1)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # out = lstm_out * alpha
        # out = F.relu(torch.sum(out, 1))
        #  x = F.relu(self.linear(hidden))
        hidden = hidden.view(bs, -1)
        logits = self.cls(hidden)
        return logits, hidden


def train_step(net, dataloader, optimizer, criterion):
    net.train()
    total_loss = []
    correct = 0
    total_data = 0
    for data in dataloader:
        pts, label = data
        pts, label = pts.to(device), label.to(device)
        bs = pts.shape[0]
        total_data += bs
        logits, _ = net(pts)

        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        index_pred = torch.argmax(F.softmax(logits), dim=1)

        correct += torch.sum(index_pred == label).item()
        total_loss.append(loss.item())

    return np.mean(total_loss), correct / total_data


def valid_step(net, dataloader, criterion):
    net.eval()
    total_loss = []
    correct = 0
    total_data = 0
    with torch.no_grad():
        for data in dataloader:
            pts, label = data
            pts, label = pts.to(device), label.to(device)
            bs = pts.shape[0]
            total_data += bs
            logits, _ = net(pts)
            loss = criterion(logits, label)

            index_pred = torch.argmax(F.softmax(logits), dim=1)
            correct += torch.sum(index_pred == label).item()
            total_loss.append(loss.item())

    return np.mean(total_loss), correct / total_data


num_frame = 100
num_points = 70 #83
# z = 50
c = 6
epochs = 30
lr = 0.0001
weight_decay = 0.0001


data_path = '../dataset.pkl'
dataset = np.load(data_path, allow_pickle=True)
subject = np.array(dataset["data"])
labels = np.array(dataset["labels"])
#subject[:, 0, :] = np.zeros((606,249))
subject_ = np.reshape(subject, (606*100 , 249))
m = subject_.mean(0)
std = subject_.std(0)

subject = (subject_-m)/std
subject = np.reshape(subject, (606, 100 , 249))
# data_pro = []
# for ani in subject:
#     sbj = []
#     for frame in ani:
#         if frame.max()-frame.min() == 0:
#             mtx2 = frame
#         else:
#             mtx1, mtx2, disparity = procrustes(m, frame)
#         sbj.append(mtx2)

#     data_pro.append(np.array(sbj))
# subject = np.array(data_pro)
# print(subject.shape)
# subject = np.reshape(subject, (60600,83, 3))
# subject = (subject-subject.mean(0))/subject.std(0)
# subject = np.reshape(subject, (606, 100,83, 3))

#subject = np.reshape(subject, (606,100,249))
random_seeds = 2021
test_size = 0.2
path = "classifier/"
if not os.path.exists(path):
    os.mkdir(path)

net = LSTMClassifier(3*num_points, num_frame, c).to(device)
criterion = nn.CrossEntropyLoss()
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state = random_seeds)
n = 0
results = []


for train_index, test_index in skf.split(subject, labels):
    n+=1

    train, valid = train_test_split(train_index, test_size=test_size, random_state = random_seeds)

    train_data = subject[train]
    train_labels = labels[train]

    print("number of train data:", train_data.shape[0])
    #mean, std, train_data = normalize(train_data)
    train_dataset = FaceData(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    valid_data = subject[valid]
    valid_labels = labels[valid]
    print("number of valid data:", valid_data.shape[0])
   # _, _, valid_data = normalize(valid_data, mean = mean, std = std)
    valid_dataset = FaceData(valid_data, valid_labels)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
#    std = torch.FloatTensor(std).unsqueeze(0).unsqueeze(0).to(device)
    std = 1
    best_loss = 1000
    for ep in range(epochs):
        train_loss, train_acc = train_step(net,  train_loader, optimizer, criterion)
        valid_loss, valid_acc = valid_step(net,  valid_loader, criterion)

        print("epoch {}, training: acc {}, loss {}".format(ep,   train_acc,  train_loss))
        print(" validation: acc {}, loss {}".format(   valid_acc, valid_loss))
        if best_loss > valid_loss:

            torch.save(net, 'classfier_'+str(n)+ '.pt')

    test_dataset = torch.load( 'classfier_'+str(n)+ '.pt')
    test_data = subject[test_index]
    test_labels = labels[test_index]
    print("number of test data:", test_data.shape[0])
   # _, _, valid_data = normalize(valid_data, mean = mean, std = std)
    test_dataset = FaceData(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loss, test_acc = valid_step(net,  test_loader, criterion)
    print(" test: acc {}, loss {}".format(test_acc, test_loss))
    results.append(test_acc)
print("Final results: mean {}, std {}".format(np.mean(results), np.std(results)))
