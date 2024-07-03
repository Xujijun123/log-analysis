import os
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import f1_score, precision_score, recall_score



# 定义数据集类
class logDataset(Dataset):
    """Log Anomaly Features Dataset"""

    def __init__(self, data_vec, labels=None):
        self.X = data_vec
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data_matrix = self.X[idx]
        if self.y is not None:
            return data_matrix, self.y[idx]
        else:
            return data_matrix


# 定义模型结构
class logCNN(nn.Module):
    def __init__(self, num_classes):
        super(logCNN, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1056, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
def main(npy_file_path, csv_file_path):
    # 数据位置
    # 加载测试数据
    test_data = np.load(npy_file_path)
    read_test_labels = pd.read_csv(csv_file_path)
    test_labels = (read_test_labels['Label'] == 'Anomaly').astype(int)

    # 数据预处理
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_data = F.pad(input=test_data, pad=(1, 1, 1, 1), mode='constant', value=0)
    test_data = np.expand_dims(test_data, axis=1)

    # 加载数据集
    test_dataset = logDataset(test_data, labels=test_labels)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)


    # 设备选择
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 初始化并加载模型
    model = logCNN(num_classes=2)
    model.load_state_dict(torch.load('model/log-anomaly/model/best_model.pth'))
    model.to(DEVICE)


    # Evaluate metrics on the test set
    y_hats = []
    y_acts = []
    counter = 0
    for i, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(DEVICE)
        yhat = model(inputs)[-1].cpu().detach().numpy().round()
        yhat = np.argmax(yhat, axis=1)
        y_hats.append(yhat)
        y_acts.append(list(targets.cpu().detach().numpy()))
        counter += 1

    y_hats = [item for sublist in y_hats for item in sublist]
    y_acts = [item for sublist in y_acts for item in sublist]
    save_location = os.path.dirname(npy_file_path)

    print("TEST SET METRICS:")
    f1 = f1_score(y_acts, y_hats)
    print("f1 score : ", f1)
    precision = precision_score(y_acts, y_hats)
    print("precision", precision)
    recall = recall_score(y_acts, y_hats)
    print("recall", recall)

    test_ys = pd.DataFrame(list(zip(y_acts, y_hats)), columns=["y_true", "y_pred"])

    print("TEST SET:\n")
    print("anomalous:\n")
    test_anomalous = test_ys[test_ys["y_true"]==1]
    print("number of anomalies in the test set:", len(test_anomalous))
    correct_anomalous = test_anomalous[test_anomalous["y_true"] == test_anomalous["y_pred"]]
    print("number of anomalies correctly identified", len(correct_anomalous))
    incorrect_anomalous = test_anomalous[test_anomalous["y_true"] != test_anomalous["y_pred"]]
    print("number of anomalies incorrectly identified", len(incorrect_anomalous))

    print("\nnormal:\n")
    test_normals = test_ys[test_ys["y_true"]==0]
    print("number of normals in the test set:", len(test_normals))
    correct_normal = test_normals[test_normals["y_true"] == test_normals["y_pred"]]
    print("number of normals correctly identified", len(correct_normal))
    incorrect_normal = test_normals[test_normals["y_true"] != test_normals["y_pred"]]
    print("number of normals incorrectly identified", len(incorrect_normal))

    #test_anomalous.to_csv('{}/anomalous_lines.csv'.format(save_location), index=True)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python test.py <npy_file_path> <csv_file_path>")
        sys.exit(1)

    npy_file_path = sys.argv[1]
    csv_file_path = sys.argv[2]

    main(npy_file_path, csv_file_path)