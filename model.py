import pickle5 as pickle
import torch
import torch.nn as nn
from torch.nn import functional
import torch.optim as optim
import argparse
from collections import Counter

class Dataset():

    def __init__(self, path_dataset):
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

        # each row corresponds to a frame. Now we need to group all frame data of one video together. Each video should have only one label
        # 047_001_001_18.jpg
        videos = list(Counter([i[:11] for i in idxs]))
        videos.sort()
        videos = videos[1:]

        for video_name in videos:
            labels.append(video_name)
            # find all frames of the same video, group them together. That's an instance
            frame_names = [i for i in idxs if video_name in i]
            frame_names.sort()  # to make sure an instance is composed of [frame1,frame2,frame3...] in stead of random order
            frame_idxs = [i for i, x in enumerate(idxs) if x in frame_names]
            video_feature = list()
            for frame_idx in frame_idxs:
                video_feature.append(features[frame_idx])
            instances.append(video_feature)

        targets = self.create_targets(labels)
        signer = self.create_signer_list(labels)

        self.signers = signer
        self.labels = targets
        self.instances = instances

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        instance = self.instances[idx]
        sample = {"Data": instance, "Class": label}

        return sample

    def create_test_set(self):
        # signer 10 is used as test set
        idx_signer10 = [i for i, x in enumerate(self.signers) if x == 10]
        self.test_x = [self.instances[idx] for idx in idx_signer10]
        self.test_y = [self.labels[idx] for idx in idx_signer10]

    def train_dev_split(self, dev):
        # dev is the ID of the signer to be used as dev set.
        idx_signer_dev = [i for i, x in enumerate(self.signers) if x == dev]
        self.dev_x = [self.instances[idx] for idx in idx_signer_dev]
        self.dev_y = [self.labels[idx] for idx in idx_signer_dev]

        idx_signer_train = [i for i, x in enumerate(self.signers) if x not in [dev, 10]]
        self.train_x = [self.instances[idx] for idx in idx_signer_train]
        self.train_y = [self.labels[idx] for idx in idx_signer_train]
        #这里得加个shuffle

    def create_targets(self, idx):
        y = list()
        for i in idx:
            video = i[:3]
            video_int = int(video.lstrip("0"))
            y.append(video_int)
        return y

    def create_signer_list(self, idx):
        signers = list()
        for i in idx:
            signer = i[4:7]
            signer_int = int(signer.lstrip("0"))
            signers.append(signer_int)
        return signers


class sign_translator(nn.Module):

    def __init__(self, hidden_size,output_size, bidirectional=True, num_layers=2, dropout=0.2):

        super(sign_translator, self).__init__()
        self.layer1 = nn.LSTM(48,hidden_size,bidirectional = bidirectional,num_layers=num_layers,dropout=dropout)      #The LSTM layer
        self.layer2 = nn.Linear(hidden_size*2 if bidirectional == True else hidden_size,output_size)  #The linear layer
        self.layer3 = nn.Softmax()   #The output layer with softmax

    def forward(self, vectors):
        # reshape the input for LSTM layer. The size of the expected input is [sequence length x 1 x 48]
        video_input = torch.reshape(vectors, [vectors.shape[0], 1, 48])#这里要改
        output_layer1, (hidden, cell) = self.layer1(video_input )
        output_layer2 = self.layer2(output_layer1)
        prediction = self.layer3(output_layer2)
        print(prediction)

        return prediction


def train_model(model,x,y,optimizer,loss_function):
    print("-------------------------------Training-------------------------------------------")
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    data_num = len(x)
    print(len(x))
    print(len(x[0]))
    print(len(y))
    print(len(y[0]))
    for train_x,train_y in zip(x,y):
        pass
        print(x)
        print(y)
        break
        optimizer.zero_grad()
        model_prediction = model(train_x)
        #这里肯定要改
        print(model_prediction)
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


def evaluate_model(model, x,y, loss_function):
    print("------------------------------------Evaluation---------------------------------------------")
    model.eval()
    epoch_loss = 0
    epoch_accuracy = 0
    data_num = len(x)
    with torch.no_grad():
        for dev_x,dev_y in zip(x,y):
            model_prediction = model(dev_x)
            model_prediction = torch.reshape(model_prediction, [model_prediction.shape[0], model_prediction.shape[2]])
            loss = loss_function(model_prediction, dev_y)
            epoch_accuracy += calculate_accuracy_per_batch(model_prediction, dev_y)
            epoch_loss += loss.item()

        accuracy = epoch_accuracy / data_num
        loss = epoch_loss / data_num
        print(f"The averaged loss is {loss}")
        print(f"The averaged accuracy is {accuracy}")


def cross_val(pathDataset,lr= 0.0001):
    with open(pathDataset, 'rb') as inp:
        dataset = pickle.load(inp)
    dataset.create_test_set()
    loss_function = nn.functional.cross_entropy
    for i in range(1,10):
        print(f"--------------Batch {i}---------------")
        model = sign_translator(hidden_size=64, output_size=64)
        optimizer = optim.Adam(params=model.parameters(), lr=lr)
        dataset.train_dev_split(i) #i_th signer for dev set, 10th signer for test set, the rest for train set
        train_model(model, dataset.train_x,dataset.train_y, optimizer, loss_function)
        evaluate_model(model, dataset.dev_x,dataset.dev_y,loss_function)

    print("--------------Final Evaluation---------------")
    evaluate_model(model, dataset.test_x,dataset.test_y, loss_function)


if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathDataset", help="path to the pickled dataset object")
    args = a.parse_args()
    cross_val(args.pathDataset)