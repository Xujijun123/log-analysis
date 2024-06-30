import pandas as pd
import numpy as np
import logging
# count rows
with open('project_raw/HDFS.log', "r") as file:
    totaln=0
    for line in file:
        totaln += 1
print('There are a total of {} lines'.format(totaln))

# Take a quick look at the data:

data = []
with open('project_raw/HDFS.log', "r") as file:
    n = 0
    for line in file:
        data.append(line)
        if n < 200:
            n += 1
        else:
            break

df = pd.DataFrame(data)
df.head()
df.iloc[100:120].values

train_idx = int(totaln*.8)
train_idx
# read the training lines only
train_data = []
with open('project_raw/HDFS.log', "r") as file:
    n=0
    for line in file:
        if n < train_idx:
            train_data.append(line)
            n += 1
        else: break
# write to the training file `HDFS_train.log`
# with open('project_raw/HDFS_train.log', 'x') as file:
#     for i in train_data:
#         file.write(i)

all_parsed = pd.read_csv('project_parsed/HDFS.log_structured.csv')

test_parsed = all_parsed.iloc[train_idx:]

test_parsed.to_csv('project_parsed/HDFS_test.log_structured.csv', index = False)