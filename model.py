import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional
from torch.nn import Module
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_data(pathH,pathB):
    hand = pd.read_csv(pathH,sep=",") #each frame correspond to a 64-dimensional vector
    body = pd.read_csv(pathB,sep=",").iloc[:,45:69] #delete keypoints related to face and hand. Each frame correspond to a 24-dimensional vector

    #normalization
    pca = PCA(n_components=24)
    hand_standard = StandardScaler().fit_transform(hand.iloc[:, 1:])
    hand_24 = pca.fit_transform(hand_standard)
    body_24 = StandardScaler().fit_transform(body)

    #split hand data and body data into train, dev, and test
    #I want to use videos of subject 7 as dev set and 8 as test set.
    body_dev = get_rows(body_24, "_007_")
    hand_dev = get_rows(hand_24, "_007_")
    body_test = get_rows(body_24, "_008_")
    hand_test = get_rows(hand_24, "_008_")

    hand_train = does_not_contains(hand_24, "_007_|_008_")
    body_train = does_not_contains(body_24, "_008_|_007_")

    #create dataloader
    #9 dataloaders in total: hand_train_loader, hand_dev_loader, hand_test_loader, body_train_loader, body_dev_loader,body_test_loader,hb_train_loader,hb_dev_loader,hb_test_loader.





class sign_translator(nn.Module):

    def __init__(self, hidden_size, bidirectional=True, num_layers=1, dropout=0):
        super(sign_translator, self).__init__()
        self.layer1 = nn.LSTM(24, hidden_size, bidirectional=bidirectional, num_layers=num_layers,
                              dropout=dropout)  # The LSTM layer
        self.layer3 = nn.Softmax(hidden_size, 64)

    def forward(self, vectors):
        # reshape the input for LSTM layer. The size of the expected input is [sequence length x 1 x 24]
        embedding_input = torch.reshape(vectors, [vectors.shape[0], 1, 24])
        output_layer1, (hidden, cell) = self.layer1(vectors)
        prediction = self.layer2(output_layer1)

        return prediction


def train_model(model,train_data,optimizer,loss_function):
    print("-------------------------------Training-------------------------------------------")
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    data_num = len(train_data)

    for batch in train_data:
        optimizer.zero_grad()
        train_x = batch[0][0]
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


def cross_val():
    pass