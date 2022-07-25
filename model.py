from collections import Counter
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional
from torch.nn import Module
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
class Dataset():

    def __init__(self,path_dataset):
        instances = list()
        labels = list()
        f = open(path_dataset)
        data = f.read()
        rows = data.split("\n")
        idxs = list()
        features = list()
        for row in rows:
            row_data = row.split(",")
            idxs.append(row_data[0])
            features.append(row_data[1:])

        #each row corresponds to a frame. Now let's group all frame data of one video together. Each video should have only one label
        #047_001_001_18.jpg
        videos = list(Counter([i[:11] for i in idxs]))
        videos.sort()
        for video_name in videos:
            labels.append(video_name)
            #find all frames of the same video, group them together. That's an instance
            frame_names = [i for i in idxs if video_name in i]
            frame_names.sort()#to make sure an instance is composed of [frame1,frame2,frame3...] in stead of random order
            frame_idxs = [i for i, x in enumerate(idxs) if x in frame_names]
            video_feature = list()
            for frame_idx in frame_idxs:
                video_feature.append(features[frame_idx])
            instances.append(video_feature)

        self.labels = labels
        self.instances = instances

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        label = self.labels[idx]
        instance = self.instances[idx]
        sample = {"Data": instance, "Class": label}

        return sample

    def test_set(self):
        #还没写好
        self.test_y = y
        self.test_x = x

    def train_dev_set(self,dev):
        #dev is the ID of the signer to be used as dev set.


def create_targets(df):
    idx = list(df.iloc[:, 0])
    y = list()
    for i in idx:
        y.append(int(i[:3]))

    return y



class sign_translator(nn.Module):

    def __init__(self, hidden_size,output_size, bidirectional=True, num_layers=2, dropout=0.2):

        super(sign_translator, self).__init__()
        self.layer1 = nn.LSTM(48,hidden_size,bidirectional = bidirectional,num_layers=num_layers,dropout=dropout)      #The LSTM layer
        self.layer2 = nn.Linear(hidden_size*2 if bidirectional == True else hidden_size,output_size)  #The linear layer
        self.layer3 = nn.Softmax(output_size,64)   #The output layer with softmax

    def forward(self, vectors):
        # reshape the input for LSTM layer. The size of the expected input is [sequence length x 1 x 24]
        video_input = torch.reshape(vectors, [vectors.shape[0], 1, 48])
        output_layer1, (hidden, cell) = self.layer1(video_input )
        output_layer2 = self.layer2(output_layer1)
        prediction = self.layer3(output_layer2)


        return prediction


def train_model(model,train_data,optimizer,loss_function):
    print("-------------------------------Training-------------------------------------------")
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    data_num = len(train_data)

    for batch in train_data:
        optimizer.zero_grad()
        train_x = batch[0][0]#这里要改
        train_y = batch[0][1]
        model_prediction = model(train_x)
        model_prediction = torch.reshape(model_prediction, [model_prediction.shape[0], model_prediction.shape[2]])
        loss_per_batch = loss_function(model_prediction, train_y)
        epoch_accuracy += calculate_accuracy_per_batch(model_prediction, train_y)
        epoch_loss += loss_per_batch.item()

        loss_per_batch.backward()
        optimizer.step()

    accuracy = epoch_accuracy / data_num
    loss = epoch_loss / data_num
    print(f"The averaged loss per instance is {loss}")
    print(f"The averaged accuracy per instance is {accuracy}")

def calculate_accuracy_per_batch(prediction,y):

    prediction = torch.max(prediction,1)[1]
    correct = 0
    for i,j in zip(prediction,y):
        if i==j:
            correct+=1
    accuracy_per_batch = correct/len(y)

    return accuracy_per_batch


def evaluate_model(model, dev_data, loss_function):
    print("------------------------------------Evaluation---------------------------------------------")
    model.eval()
    epoch_loss = 0
    epoch_accuracy = 0
    data_num = len(dev_data)
    with torch.no_grad():
        for batch in dev_data:
            dev_x = batch[0][0]
            dev_y = batch[0][1]
            model_prediction = model(dev_x)
            model_prediction = torch.reshape(model_prediction, [model_prediction.shape[0], model_prediction.shape[2]])
            loss = loss_function(model_prediction, dev_y)
            epoch_accuracy += calculate_accuracy_per_batch(model_prediction, dev_y)
            epoch_loss += loss.item()

        accuracy = epoch_accuracy / data_num
        loss = epoch_loss / data_num
        print(f"The averaged loss is {loss}")
        print(f"The averaged accuracy is {accuracy}")


def cross_val(pathDataset):
    dataset = Dataset(pathDataset)
    model = (hidden_size = 64,output_size=64)
    for i in range(1,10):
        dev_data = dataset.extract_signer(i)
        train_data = dataset.exclude_signer([i,10])#signer 10 is for test set

    train_model(model, train_data, optimizer, loss_function)
    evaluate_model(model, dev_data, loss_function)