import pickle5 as pickle
import torch
import torch.nn as nn
from torch.nn import functional
import torch.optim as optim
import argparse
from collections import Counter
import sklearn
from sklearn.utils import shuffle

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
        self.labels = [i-1 for i in targets]
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
        dev_x = [self.instances[idx] for idx in idx_signer_dev]
        dev_y = [self.labels[idx] for idx in idx_signer_dev]
        self.dev_x,self.dev_y = sklearn.utils.shuffle(dev_x,dev_y)

        idx_signer_train = [i for i, x in enumerate(self.signers) if x not in [dev, 10]]
        train_x = [self.instances[idx] for idx in idx_signer_train]
        train_y = [self.labels[idx] for idx in idx_signer_train]
        self.train_x, self.train_y= sklearn.utils.shuffle(train_x,train_y)

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

    def __init__(self,bidirectional=True, num_layers=2, dropout=0.2):
        super(sign_translator, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=10,stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=10, kernel_size=10,stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=10, out_channels=1, kernel_size=7,stride=2)
        self.lstm = torch.nn.LSTM(input_size=6,hidden_size=128,num_layers=2)
        self.linear = torch.nn.Linear(128, 64)

    def forward(self,inputs):
        output1 = self.conv1(inputs)
        output2 = self.conv2(output1)
        output3 = self.conv3(output2)
        video = output3.reshape(output3.size()[0],output3.size()[3])
        output_layer1, (hidden, cell) = self.lstm(video)
        last_output = output_layer1[-1]
        prediction = self.linear(last_output)

        return prediction


def train_model(model,x,y,optimizer,loss_function):
    print("-------------------------------Training-------------------------------------------")
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    data_num = len(x)
    for train_x,train_y in zip(x,y):
        train_x = torch.stack(train_x)
        train_y = torch.tensor(train_y)
        train_y = torch.nn.functional.one_hot(train_y, num_classes=64)
        train_y = train_y.type(torch.FloatTensor)
#        train_y = torch.reshape(train_y, [1])
        optimizer.zero_grad()
        model_prediction = model(train_x)
        loss_per_batch = loss_function(model_prediction, train_y)
        epoch_accuracy += correct_or_not(model_prediction, train_y)
        epoch_loss += loss_per_batch.item()

        loss_per_batch.backward()
        optimizer.step()

    accuracy = epoch_accuracy / data_num
    loss = epoch_loss / data_num
    print(f"The averaged loss per instance is {loss}")
    print(f"The averaged accuracy per instance is {accuracy}")

def correct_or_not(prediction,y):
    prediction = torch.max(prediction,0)[1]
    target = torch.max(y,0)[1]
    if prediction ==target:
        return 1
    else:
        return 0


def evaluate_model(model, x,y, loss_function):
    print("------------------------------------Evaluation---------------------------------------------")
    model.eval()
    epoch_loss = 0
    epoch_accuracy = 0
    data_num = len(x)
    with torch.no_grad():
        for dev_x,dev_y in zip(x,y):
            dev_x = torch.stack(dev_x)
            dev_y = torch.tensor(dev_y)
            dev_y = torch.nn.functional.one_hot(dev_y, num_classes=64)
            dev_y = dev_y.type(torch.FloatTensor)
            model_prediction = model(dev_x)
            loss = loss_function(model_prediction, dev_y)
            epoch_accuracy += correct_or_not(model_prediction, dev_y)
            epoch_loss += loss.item()

        accuracy = epoch_accuracy / data_num
        loss = epoch_loss / data_num
        print(f"The averaged loss is {loss}")
        print(f"The averaged accuracy is {accuracy}")


def cross_val(pathDataset,lr= 0.001):
    with open(pathDataset, 'rb') as inp:
        dataset = pickle.load(inp)
    dataset.create_test_set()
    loss_function = nn.functional.cross_entropy
    for i in range(1,10):
        print(f"--------------Epoch {i}---------------")
        model = sign_translator()
        optimizer = optim.Adam(params=model.parameters(), lr=lr)
        dataset.train_dev_split(i) #i_th signer for dev set, 10th signer for test set, the rest for train set
        train_model(model, dataset.train_x,dataset.train_y, optimizer, loss_function)
        evaluate_model(model, dataset.dev_x,dataset.dev_y,loss_function)

    print("--------------Final Evaluation---------------")
    evaluate_model(model, dataset.test_x,dataset.test_y, loss_function)


if __name__=="__main__":
#    a = argparse.ArgumentParser()
#    a.add_argument("--pathDataset", help="path to the pickled dataset object")
#    args = a.parse_args()
    cross_val("/home/CE/zhangshi/signlanguage/image_dataset.pickle")