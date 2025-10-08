import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loaders.data_loader_utils import FAMOS_EXPRESSION_LIST, COMA_EXPRESSION_LIST
from utils.fixseed import fixseed
from utils.parser_util import classifier_args, train_args
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
# from train.training_loop import TrainLoop
from train.train_platforms import ClearmlPlatform, WandBPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
import os
os.system('export clearml_log_level=ERROR')
from sklearn.metrics import confusion_matrix
# os.environ['MPLCONFIGDIR'] = "/home/dcor/yaronaloni/.config/matplotlib"

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay



class LSTMClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm_1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        # self.lstm_2 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.cls = nn.Linear(hidden_size * 2, num_classes)
        # self.cls = nn.Linear(hidden_size * 2, int(hidden_size/2))
        # self.cls2 = nn.Linear(int(hidden_size/2), num_classes)
        self.tanh1 = nn.Tanh()

    def forward(self, x):
        bs = x.shape[0]

        x, (hidden, cell) = self.lstm_1(x)
        # x, (hidden, cell) = self.lstm_2(x)


        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = hidden.view(bs, -1)
        logits = self.cls(hidden)
        # hidden = self.cls(hidden)
        # logits = self.cls2(hidden)
        return logits, hidden


def train_step(net, dataloader, optimizer, criterion):
    net.train()
    total_loss = []
    correct = 0
    total_data = 0
    for data, cond in dataloader:
        pts, label = data.squeeze(2).permute(0,2,1), cond['y']['action'].squeeze(1)
        pts, label = pts.to(device), label.to(device)
        bs = pts.shape[0]
        total_data += bs
        logits, _ = net(pts)

        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # index_pred = torch.argmax(F.softmax(logits), dim=1)
        index_pred = torch.argmax(F.softmax(logits, dim=1), dim=1)

        correct += torch.sum(index_pred == label).item()
        total_loss.append(loss.item())
        # cm = confusion_matrix(label.cpu().numpy(), index_pred.cpu().numpy())
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=FAMOS_EXPRESSION_LIST)
        # disp.plot()
        # plt.show()


    return np.mean(total_loss), correct / total_data


def valid_step(net, dataloader, criterion, epoch, save_dir, best_acc = 0):
    net.eval()
    total_loss = []
    correct = 0
    total_data = 0
    cm = np.zeros((len(data_list), len(data_list)))
    with torch.no_grad():
        for data, cond in dataloader:
            pts, label = data.squeeze(2).permute(0, 2, 1), cond['y']['action'].squeeze(1)
            pts, label = pts.to(device), label.to(device)
            bs = pts.shape[0]
            total_data += bs
            logits, _ = net(pts)
            loss = criterion(logits, label)

            # index_pred = torch.argmax(F.softmax(logits), dim=1)
            index_pred = torch.argmax(F.softmax(logits, dim=1), dim=1)

            correct += torch.sum(index_pred == label).item()
            total_loss.append(loss.item())

            cm += confusion_matrix(label.cpu().numpy(), index_pred.cpu().numpy(),  labels=range((len(data_list))))
        
        if correct / total_data > best_acc:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data_list)
            plt.figure(figsize=(16, 12))  # Increase the figure size as desired

            disp.plot()
            plt.xticks(rotation='vertical')  # Rotate the x-axis labels vertically
            plt.title(f"Validation Confusion Matrix epoch: {epoch}, accuracy: {round((100*correct / total_data), 2)}")
            plt.ylabel('GT')
            plt.xlabel('Predicted')
            plt.savefig(os.path.join(save_dir, f"conf_mtrx_ep_{epoch}.png"))
            with open(os.path.join(save_dir, f"conf_mtrx_ep_{epoch}.data"), 'wb') as f:
                np.save(f, cm)

    return np.mean(total_loss), correct / total_data


if __name__ == '__main__':

    args = classifier_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.dataset == 'famos':
        data_list = FAMOS_EXPRESSION_LIST
    elif args.dataset == 'coma':
        data_list = COMA_EXPRESSION_LIST
    labels_map = {}

    for i in range(len(data_list)):
        labels_map[i] = data_list[i]

    device = dist_util.dev()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    else:
        if not args.overwrite:
            raise ValueError("Save directory already exists.")
        else:
            print("Overwriting save directory.")
            shutil.rmtree(args.save_dir)
            os.makedirs(args.save_dir)


    hidden_size = args.hidden_size
    num_points = 70 #83
    c = len(data_list)
    epochs = 1000
    lr = args.lr
    weight_decay = args.weight_decay
    normalize_data = not args.normalize_data_off
    train_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames,
                                      hml_mode='train_classifier', data_mode=args.data_mode, split='train', classifier_step=args.classifier_step, minimum_frames=args.minimum_frames, debug=args.debug, normalize_data=normalize_data)

    valid_bs = (23 if args.dataset == 'coma' else args.batch_size)

    #train_classifier is used in order to use the classifier_step
    valid_loader = get_dataset_loader(name=args.dataset, batch_size=valid_bs, num_frames=args.num_frames,
                              data_mode=args.data_mode, hml_mode='train_classifier', split='test', classifier_step=args.classifier_step, minimum_frames=args.minimum_frames, debug=args.debug, normalize_data=normalize_data)

    random_seeds = args.seed

    net = LSTMClassifier(3*num_points, hidden_size, 1, c).to(device)
    criterion = nn.CrossEntropyLoss()
    # skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=random_seeds)
    n = 0
    results = []

    if args.classifier_path!='':
        net.load_state_dict(torch.load(args.classifier_path, map_location=device))

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    std = 1
    best_loss = 1000
    best_validation_acc = 0
    for ep in range(epochs):
        train_loss, train_acc = train_step(net,  train_loader, optimizer, criterion)
        valid_loss, valid_acc = valid_step(net,  valid_loader, criterion, ep, args.save_dir, best_validation_acc)

        if valid_acc > best_validation_acc: #best_loss > valid_loss:
            torch.save(net.state_dict(), os.path.join(args.save_dir, 'classfier_'+str(ep)+ '.pt'))
            best_loss = valid_loss
            best_validation_acc = valid_acc
            n=ep
            # print("save model")
        lr_scheduler.step()
        print("epoch {}, training: acc {}, loss {}".format(ep,   train_acc*100,  train_loss))
        print("validation: acc {}, loss {}".format(valid_acc*100, valid_loss))
        print("best validation acc {}, best loss: {}".format(best_validation_acc*100, best_loss))


    # net = torch.load(os.path.join(args.save_dir, 'classfier_'+str(n)+ '.pt'))

    # test_loader = DataLoader(valid_loader.dataset, batch_size=32, shuffle=True, num_workers=4)
    # test_loss, test_acc = valid_step(net,  test_loader, criterion, epochs, args.save_dir)
    # print(" test: acc {}, loss {}".format(test_acc, test_loss))
    # results.append(test_acc)
    # print("Final results: mean {}, std {}".format(np.mean(results), np.std(results)))
