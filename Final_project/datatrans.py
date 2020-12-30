import numpy as np 
import pandas as pd 

train_df = pd.read_csv("./train.csv")
train_df.head()

data = np.array(train_df)
new_data = []

for i in range(len(data)):
    for j in range(4):
        num = j+1
        if num == data[i][1]:
            new_data.append([data[i][0]+"_"+str(num), data[i][2]])
        else:
            new_data.append([data[i][0]+"_"+str(num)])

original_train_csv = pd.DataFrame(new_data, columns=["ImageId_ClassId","EncodedPixels"])
original_train_csv.to_csv("original_train.csv", index=None)

sample_df = pd.read_csv("./sample_submission.csv")
sample_df.head()

sub = np.array(sample_df)
new_sub = []

for i in range(len(sub)):
    for j in range(4):
        num = j+1
        new_sub.append([sub[i][0]+"_"+str(num), 11])

original_sub_csv = pd.DataFrame(new_sub, columns=["ImageId_ClassId","EncodedPixels"])
original_sub_csv.to_csv("original_sample_submission.csv", index=None)