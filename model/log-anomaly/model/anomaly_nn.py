import os
import time
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
random.seed(0)

np.random.seed(0)

data_loc = "../process/project_processed_data/_tf-idf_v5/"

# Train sets
train_data = np.load('{}x_train_tf-idf_v5.npy'.format(data_loc))
read_train_labels = pd.read_csv('{}y_train_tf-idf_v5.csv'.format(data_loc))
train_labels = read_train_labels['Label'] == 'Anomaly'
train_labels = train_labels.astype(int)

# Test sets
test_data = np.load('{}x_test_tf-idf_v5.npy'.format(data_loc))
read_test_labels = pd.read_csv('{}y_test_tf-idf_v5.csv'.format(data_loc))
test_labels = read_test_labels['Label'] == 'Anomaly'
test_labels = test_labels.astype(int)

train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
train_labels = train_labels.reset_index(drop=True)
val_labels = val_labels.reset_index(drop=True)

print("train data shape")
print(train_data.shape)
print(train_labels.shape)
print("val data shape")
print(val_data.shape)
print(val_labels.shape)
print(len(test_labels))
anom = test_labels[test_labels > 0]
norm = test_labels[test_labels == 0]
print(anom.shape)
print(norm.shape)

print(len(anom)/len(test_labels))
print(len(train_labels))
anom = train_labels[train_labels > 0]
norm = train_labels[train_labels == 0]
print(anom.shape)
print(norm.shape)

print(len(anom)/len(train_labels))


class logDataset(Dataset):
    """Log Anomaly Features Dataset"""

    def __init__(self, data_vec, labels=None):
        self.X = data_vec
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data_matrix = self.X[idx]

        if not self.y is None:
            return (data_matrix, self.y[idx])
        else:
            return data_matrix
# add 1 1 1 1 for padding
train_data = torch.tensor(train_data, dtype=torch.float32)
train_data = F.pad(input=train_data, pad=(1, 1, 1, 1), mode='constant', value=0) # pad all sides with 0s
train_data = np.expand_dims(train_data, axis=1)
test_data = torch.tensor(test_data, dtype=torch.float32)
test_data = F.pad(input=test_data, pad=(1, 1, 1, 1), mode='constant', value=0)
test_data = np.expand_dims(test_data, axis=1)
val_data = torch.tensor(val_data, dtype=torch.float32)
val_data = F.pad(input=val_data, pad=(1, 1, 1, 1), mode='constant', value=0)
val_data = np.expand_dims(val_data, axis=1)

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 10

NUM_CLASSES = 2

# Other
if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

# pass datasets into the custom dataclass
train_dataset = logDataset(train_data, labels = train_labels)
test_dataset = logDataset(test_data, labels = test_labels)
val_dataset = logDataset(val_data, labels = val_labels)

# use DataLoader class
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=0,  # couldn't use workers https://github.com/fastai/fastbook/issues/85
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         num_workers=0,
                         shuffle=False)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=BATCH_SIZE,
                        num_workers=0,
                        shuffle=False)

# Checking the dataset
for data, labels in train_loader:
    print('Matrix batch dimensions:', data.shape)
    print('Matrix label dimensions:', labels.shape)
    break

# Checking the dataset
for data, labels in test_loader:
    print('Matrix batch dimensions:', data.shape)
    print('Matrix label dimensions:', labels.shape)
    break

# Checking the dataset
for data, labels in val_loader:
    print('Matrix batch dimensions:', data.shape)
    print('Matrix label dimensions:', labels.shape)
    break

# Check that the train loader works correctly
device = torch.device(DEVICE)
torch.manual_seed(0)

for epoch in range(2):

    for batch_idx, (x, y) in enumerate(train_loader):
        print('Epoch:', epoch + 1, end='')
        print(' | Batch index:', batch_idx, end='')
        print(' | Batch size:', y.size()[0])

        x = x.to(device)
        y = y.to(device)
        break


##########################
### MODEL
##########################


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

torch.manual_seed(RANDOM_SEED)

model = logCNN(NUM_CLASSES)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE, dtype=torch.long)  # had to pass in to torch.long based on the 1, 0 int labels

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


# This needs to be vectorized
def compute_f1(model, data_loader, device):
    y_hats = []
    y_acts = []
    for i, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(DEVICE)
        yhat = model(inputs)[-1].cpu().detach().numpy().round()
        yhat = np.argmax(yhat, axis=1)
        y_hats.append(yhat)
        y_acts.append(list(targets.cpu().detach().numpy()))

    y_hats = [item for sublist in y_hats for item in sublist]
    y_acts = [item for sublist in y_acts for item in sublist]
    return f1_score(y_acts, y_hats)


start_time = time.time()
minibatch_cost = []
epoch_train_performance = []
epoch_val_performance = []
train_data = torch.tensor(train_data, dtype=torch.float32).to(DEVICE)
test_data = torch.tensor(test_data, dtype=torch.float32).to(DEVICE)
val_data = torch.tensor(val_data, dtype=torch.float32).to(DEVICE)
for epoch in range(NUM_EPOCHS):

    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):

        features = features.to(DEVICE)
        targets = targets.to(DEVICE, dtype=torch.long)  # another had to use torch.long

        ### FORWARD AND BACK PROP
        logits, probas = model(features)

        cost = F.cross_entropy(logits, targets)

        optimizer.zero_grad()

        cost.backward()
        minibatch_cost.append(cost)

        ### UPDATE MODEL PARAMETERS
        optimizer.step()

        ### LOGGING
        if not batch_idx % 50:
            print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                  % (epoch + 1, NUM_EPOCHS, batch_idx,
                     len(train_loader), cost,))
    best_val_f1=0
    model.eval()
    with torch.set_grad_enabled(False):  # save memory during inference
        train_performance = compute_f1(model, train_loader, device=DEVICE)
        val_performance = compute_f1(model, val_loader, device=DEVICE)
        epoch_train_performance.append(train_performance)
        epoch_val_performance.append(val_performance)
        print('Epoch: %03d/%03d | Train: %.3f%%   | Val: %.3f%%' % (
            epoch + 1, NUM_EPOCHS, train_performance, val_performance))

        if val_performance > best_val_f1:
            best_val_f1 = val_performance
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model saved!')
            
    print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

# plot cost functions
minibatch_cost_cpu = [i.cpu().detach().numpy() for i in minibatch_cost]

plt.plot(range(len(minibatch_cost_cpu)), minibatch_cost_cpu)
plt.ylabel('Cost Function Label')
plt.xlabel('Minibatch')
plt.show()

plt.plot(range(len(epoch_train_performance)), epoch_train_performance, label="train f1 scores")
plt.plot(range(len(epoch_val_performance)), epoch_val_performance, label="val f1 scores")
plt.ylabel('F1 Score')
plt.xlabel('Epoch')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

# Evaluate metrics on the test set
y_hats = []
y_acts = []
counter = 0
for i, (inputs, targets) in enumerate(test_loader):
    inputs = inputs.to(device)
    yhat = model(inputs)[-1].cpu().detach().numpy().round()
    yhat = np.argmax(yhat, axis=1)
    y_hats.append(yhat)
    y_acts.append(list(targets.cpu().detach().numpy()))
    counter += 1

y_hats = [item for sublist in y_hats for item in sublist]
y_acts = [item for sublist in y_acts for item in sublist]

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

# Evaluate metrics on train set
y_hats = []
y_acts = []
counter = 0
for i, (inputs, targets) in enumerate(train_loader):
    inputs = inputs.to(device)
    yhat = model(inputs)[-1].cpu().detach().numpy().round()
    yhat = np.argmax(yhat, axis=1)
    y_hats.append(yhat)
    y_acts.append(list(targets.cpu().detach().numpy()))
    counter += 1

y_hats = [item for sublist in y_hats for item in sublist]
y_acts = [item for sublist in y_acts for item in sublist]

print("TRAIN SET METRICS:")
f1 = f1_score(y_acts, y_hats)
print("f1 score : ", f1)
precision = precision_score(y_acts, y_hats)
print("precision", precision)
recall = recall_score(y_acts, y_hats)
print("recall", recall)

train_ys = pd.DataFrame(list(zip(y_acts, y_hats)), columns=["y_true", "y_pred"])
print("TRAIN SET:\n")
print("anomalous:\n")
train_anomalous = train_ys[train_ys["y_true"]==1]
print("number of anomalies in the train set:", len(train_anomalous))
correct_anomalous = train_anomalous[train_anomalous["y_true"] == train_anomalous["y_pred"]]
print("number of anomalies correctly identified", len(correct_anomalous))
incorrect_anomalous = train_anomalous[train_anomalous["y_true"] != train_anomalous["y_pred"]]
print("number of anomalies incorrectly identified", len(incorrect_anomalous))

print("\nnormal:\n")
train_normals = train_ys[train_ys["y_true"]==0]
print("number of normals in the train set:", len(train_normals))
correct_normal = train_normals[train_normals["y_true"] == train_normals["y_pred"]]
print("number of normals correctly identified", len(correct_normal))
incorrect_normal = train_normals[train_normals["y_true"] != train_normals["y_pred"]]
print("number of normals incorrectly identified", len(incorrect_normal))